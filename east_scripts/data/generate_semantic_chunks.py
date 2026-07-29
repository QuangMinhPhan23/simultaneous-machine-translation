"""LLM-based semantic chunker for the Egyptian Arabic training data.

An instruction-tuned model (--chunker_model llama | gemma | nilechat) splits each
English/Arabic pair into aligned chunks at 3 granularities (low / medium / high latency) in
one call, returned as JSON. Output feeds build_arabic_simt_sft_data.py --chunks_path.
If the model's chunks do not rebuild the original sentence we fall back to the word-count
split and record the reason, so the fallback rate stays visible.
"""
import argparse
import difflib
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_arabic_simt_sft_data import DIALECT_NAMES, LATENCY_CHUNK_WORDS, align_chunks, chunk_sentence_pair

CHUNKER_MODELS = {
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma": "google/gemma-3-4b-it",
    "nilechat": "MBZUAI-Paris/Nile-Chat-4B",
}

DEFAULT_MIN_RECONSTRUCTION_SIMILARITY = 0.97

# JSON key the model replies with -> the short latency label used everywhere else.
LATENCY_JSON_KEYS = {"low": "low_latency", "medium": "medium_latency", "high": "high_latency"}

CHUNK_PROMPT = """You are helping prepare training data for a simultaneous interpreter.

Given an English sentence and its {tgt_lang} translation, split BOTH sentences into aligned chunks at THREE different granularities, one per latency level:
- low latency: fragment into brief, coherent phrases that each convey one complete thought -- the MOST chunks
- medium latency: longer chunks, roughly clause-length -- fewer chunks than low latency
- high latency: the longest chunks, covering full clauses or the whole sentence -- the FEWEST chunks (often just one)

For EACH of the three granularities independently:
- each chunk is a point where a real-time interpreter could plausibly start translating without waiting for more of the source sentence
- concatenating the chunks in order EXACTLY reconstructs each original sentence, character for character -- copy substrings verbatim, do not paraphrase, correct, translate, add, remove, or reorder anything
- the English chunks and the {tgt_lang} chunks must have the SAME NUMBER of chunks, in the same order (chunk i of the source aligns with chunk i of the target)
- the chunk COUNT must not increase from low to medium to high latency (low >= medium >= high)

Example:
English: Good morning, I need ten kilograms of tomatoes for the restaurant.
{tgt_lang}: صباح الخير، محتاج عشرة كيلو طماطم للمطعم.
{{
  "low_latency": {{"source_chunks": ["Good morning,", "I need", "ten kilograms of tomatoes", "for the restaurant."], "target_chunks": ["صباح الخير،", "محتاج", "عشرة كيلو طماطم", "للمطعم."]}},
  "medium_latency": {{"source_chunks": ["Good morning, I need", "ten kilograms of tomatoes for the restaurant."], "target_chunks": ["صباح الخير، محتاج", "عشرة كيلو طماطم للمطعم."]}},
  "high_latency": {{"source_chunks": ["Good morning, I need ten kilograms of tomatoes for the restaurant."], "target_chunks": ["صباح الخير، محتاج عشرة كيلو طماطم للمطعم."]}}
}}

Now do the same for:
English: {source}
{tgt_lang}: {target}

Respond with ONLY a JSON object of this exact form, no other text:
{{"low_latency": {{"source_chunks": ["...", "..."], "target_chunks": ["...", "..."]}}, "medium_latency": {{"source_chunks": ["...", "..."], "target_chunks": ["...", "..."]}}, "high_latency": {{"source_chunks": ["...", "..."], "target_chunks": ["...", "..."]}}}}"""


def build_fallback_chunks_all_latencies(source, target):
    """Word-count fallback used when the model's response fails validation, chunked
    separately for each latency level."""
    return {
        latency: chunk_sentence_pair(source, target, words)
        for latency, words in LATENCY_CHUNK_WORDS.items()
    }


def _repair_json_brackets(candidate):
    """Fixes the two ways these models break their JSON: leaving off the outermost closing
    brace, and closing an array with '}' instead of ']'. Walks the text with a bracket
    stack, skipping string literals so brackets inside the chunk text are not counted."""
    stack = []
    in_string = False
    escape = False
    repaired = list(candidate)
    for i, ch in enumerate(repaired):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            expected_close = "}" if stack[-1] == "{" else "]"
            if ch != expected_close:
                repaired[i] = expected_close
            stack.pop()
    for opener in reversed(stack):
        repaired.append("}" if opener == "{" else "]")
    return "".join(repaired)


def extract_json_object(text):
    """Grabs the first {...} span, since model output often wraps the JSON in markdown
    fences or chatter. If that does not parse, retries once on a bracket-repaired copy."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_repair_json_brackets(candidate))
    except json.JSONDecodeError:
        return None
    return None


def normalize_for_comparison(text):
    """NFC-normalizes and drops spaces before comparing, so Arabic hamza/diacritic encoding
    variants do not look like real content differences."""
    return unicodedata.normalize("NFC", text).replace(" ", "")


def reconstruction_similarity(chunks, original):
    joined = normalize_for_comparison("".join(chunks))
    original_norm = normalize_for_comparison(original)
    return difflib.SequenceMatcher(None, joined, original_norm).ratio()


def reconstructs(chunks, original, min_similarity):
    return reconstruction_similarity(chunks, original) >= min_similarity


def parse_chunk_response(response_text, source, target, min_similarity=DEFAULT_MIN_RECONSTRUCTION_SIMILARITY):
    """Validates each latency on its own, so one malformed granularity does not throw away
    the other two. Only the failing latency falls back to chunk_sentence_pair().

    Returns (chunks_by_latency, fallback_by_latency, reason_by_latency):
      chunks_by_latency[latency]   -> (source_chunks, target_chunks), always populated
      fallback_by_latency[latency] -> True if that latency used the fallback
      reason_by_latency[latency]   -> failure reason string, or None if it succeeded

    A JSON parse failure fails all three, since there is nothing to salvage."""
    parsed = extract_json_object(response_text)
    if not isinstance(parsed, dict):
        chunks_by_latency = {
            latency: chunk_sentence_pair(source, target, words) for latency, words in LATENCY_CHUNK_WORDS.items()
        }
        fallback_by_latency = {latency: True for latency in LATENCY_CHUNK_WORDS}
        reason_by_latency = {latency: "json_parse_failed" for latency in LATENCY_CHUNK_WORDS}
        return chunks_by_latency, fallback_by_latency, reason_by_latency

    chunks_by_latency, fallback_by_latency, reason_by_latency = {}, {}, {}
    for latency, json_key in LATENCY_JSON_KEYS.items():
        chunk_words = LATENCY_CHUNK_WORDS[latency]
        entry = parsed.get(json_key)
        reason = None
        source_chunks = target_chunks = None
        if not isinstance(entry, dict):
            reason = "missing_latency_key"
        else:
            source_chunks = entry.get("source_chunks")
            target_chunks = entry.get("target_chunks")
            if not isinstance(source_chunks, list) or not isinstance(target_chunks, list):
                reason = "missing_or_wrong_type_keys"
            elif not source_chunks or not target_chunks:
                reason = "empty_chunk_list"
            elif not all(isinstance(c, str) and c.strip() for c in source_chunks + target_chunks):
                reason = "invalid_chunk_strings"
            elif not reconstructs(source_chunks, source, min_similarity) or not reconstructs(target_chunks, target, min_similarity):
                reason = "reconstruction_mismatch"

        if reason is None:
            chunks_by_latency[latency] = align_chunks(source_chunks, target_chunks)
        else:
            chunks_by_latency[latency] = chunk_sentence_pair(source, target, chunk_words)
        fallback_by_latency[latency] = reason is not None
        reason_by_latency[latency] = reason

    return chunks_by_latency, fallback_by_latency, reason_by_latency


# Turn-end tokens that tokenizer.eos_token_id does not cover. Without them generate()
# never stops early and runs to max_new_tokens on every call.
_EXTRA_TURN_END_TOKENS = ["<|eot_id|>", "<end_of_turn>"]


def _resolve_eos_token_ids(tokenizer):
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    for tok in _EXTRA_TURN_END_TOKENS:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            eos_ids.add(tok_id)
    return list(eos_ids)


class ChunkGenerator:
    def __init__(self, model_path, min_similarity=DEFAULT_MIN_RECONSTRUCTION_SIMILARITY, debug_raw_responses=0):
        self.min_similarity = min_similarity
        # Prints the next N raw responses to stderr, before parsing throws the text away.
        self.debug_raw_responses = debug_raw_responses
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.eos_token_ids = _resolve_eos_token_ids(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    @torch.no_grad()
    def generate_chunks(self, source, target, tgt_lang):
        """Returns (chunks_by_latency, fallback_by_latency, reason_by_latency), see parse_chunk_response."""
        prompt = CHUNK_PROMPT.format(source=source, target=target, tgt_lang=tgt_lang)
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            # 3 granularities in one response need the headroom; eos_token_id stops most
            # turns well before the limit anyway.
            **inputs, max_new_tokens=1024, do_sample=False, num_beams=1,
            eos_token_id=self.eos_token_ids,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        response_text = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        if self.debug_raw_responses > 0:
            self.debug_raw_responses -= 1
            print(f"--- RAW RESPONSE (source={source[:60]!r}) ---\n{response_text}\n--- END RAW RESPONSE ---", file=sys.stderr)

        return parse_chunk_response(response_text, source, target, self.min_similarity)


def load_existing(output_path):
    """Loads already-finished entries for --resume. Entries that fell back to the heuristic
    are dropped so the main loop retries them instead of keeping the fallback forever."""
    if not output_path or not os.path.exists(output_path):
        return [], set()
    with open(output_path, "r", encoding="utf-8") as f:
        all_entries = json.load(f)
    entries = [e for e in all_entries if not e.get("fallback")]
    done = {(e["conv_id"], e["turn_idx"]) for e in entries}
    n_dropped = len(all_entries) - len(entries)
    if n_dropped:
        print(f"Resuming: {n_dropped} previously-fallback entries will be retried, {len(entries)} kept as-is", file=sys.stderr)
    return entries, done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunker_model", choices=list(CHUNKER_MODELS.keys()), required=True)
    parser.add_argument("--country", default="EG")
    parser.add_argument("--split", default="train")
    parser.add_argument("--direction", choices=["en2ar", "ar2en"], default="en2ar")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true",
                         help="Skip (conv_id, turn_idx) pairs already present in --output")
    parser.add_argument("--save_every", type=int, default=200,
                         help="Flush --output to disk every N newly-generated turns")
    parser.add_argument("--max_examples", type=int, default=None,
                         help="Cap the number of turns processed, for a quick sanity check")
    parser.add_argument("--min_reconstruction_similarity", type=float, default=DEFAULT_MIN_RECONSTRUCTION_SIMILARITY,
                         help="Fuzzy-match threshold (0-1) for accepting a chunking; 1.0 means "
                              "exact character match")
    parser.add_argument("--debug_raw_responses", type=int, default=0,
                         help="Print the first N raw model responses to stderr, for debugging "
                              "parse failures")
    args = parser.parse_args()

    model_path = CHUNKER_MODELS[args.chunker_model]
    print(f"Loading chunker model: {model_path}")
    generator = ChunkGenerator(model_path, min_similarity=args.min_reconstruction_similarity,
                                debug_raw_responses=args.debug_raw_responses)

    dataset = load_dataset("UBC-NLP/alexandria", name=args.country, split=args.split)
    tgt_dialect_name = DIALECT_NAMES.get(args.country, f"Arabic ({args.country})")

    entries, done = load_existing(args.output) if args.resume else ([], set())
    n_processed, n_fallback = 0, 0
    fallback_reasons = Counter()
    n_fallback_by_latency = Counter()

    for conv in dataset:
        conv_id = conv["conv_id"]
        for turn_idx, (eng_turn, dia_turn) in enumerate(zip(conv["english_conversation"], conv["dialectal_conversation"])):
            if args.max_examples and n_processed >= args.max_examples:
                break
            if (conv_id, turn_idx) in done:
                continue

            eng_text = eng_turn["text"].strip()
            dia_text = dia_turn["text"].strip()
            if not eng_text or not dia_text:
                continue

            if args.direction == "en2ar":
                source, target, src_lang, tgt_lang = eng_text, dia_text, "English", tgt_dialect_name
            else:
                source, target, src_lang, tgt_lang = dia_text, eng_text, tgt_dialect_name, "English"

            try:
                chunks_by_latency, fallback_by_latency, reason_by_latency = generator.generate_chunks(source, target, tgt_lang)
            except Exception as e:
                print(f"WARNING: generation failed for conv={conv_id} turn={turn_idx} ({e}); using heuristic fallback",
                      file=sys.stderr)
                chunks_by_latency = build_fallback_chunks_all_latencies(source, target)
                fallback_by_latency = {latency: True for latency in LATENCY_CHUNK_WORDS}
                reason_by_latency = {latency: f"generation_exception: {e}" for latency in LATENCY_CHUNK_WORDS}

            fallback = any(fallback_by_latency.values())
            entries.append({
                "conv_id": conv_id,
                "turn_idx": turn_idx,
                "chunks": {
                    latency: {"source_chunks": sc, "target_chunks": tc}
                    for latency, (sc, tc) in chunks_by_latency.items()
                },
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "fallback": fallback,
                "fallback_by_latency": fallback_by_latency,
                "fallback_reason_by_latency": reason_by_latency,
            })
            n_processed += 1
            n_fallback += int(fallback)
            for latency, reason in reason_by_latency.items():
                if reason is not None:
                    n_fallback_by_latency[latency] += 1
                    fallback_reasons[f"{latency}:{reason}"] += 1

            if n_processed % args.save_every == 0:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(entries, f, ensure_ascii=False, indent=2)
                print(f"  ...{n_processed} turns processed ({n_fallback} fallbacks so far)", file=sys.stderr)

        if args.max_examples and n_processed >= args.max_examples:
            break

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(entries)} entries to {args.output} "
        f"({n_processed} newly generated this run, {n_fallback} turns had at least one fallback latency)"
    )
    for latency in LATENCY_CHUNK_WORDS:
        print(f"  {latency}: {n_fallback_by_latency[latency]}/{n_processed} fell back to the word-count heuristic")
    if fallback_reasons:
        print("Fallback reason breakdown (this run only):", file=sys.stderr)
        for reason, count in fallback_reasons.most_common():
            print(f"  {reason}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
