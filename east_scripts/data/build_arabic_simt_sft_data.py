"""Build Egyptian Arabic SiMT + offline training data from the Alexandria dataset.

Input is the UBC-NLP/alexandria train split. Output is alpaca-style JSON with 3 SiMT
examples per sentence pair (low / medium / high latency) plus 1 offline example, written in
EAST's interleaved <|end-of-read|> / <|end-of-write|> format.
Chunk boundaries come from a word-count split, from --chunks_path, or from the whole
sentence with --granularity turn.
"""
import argparse
import json
import os
import random
import sys

from datasets import load_dataset

EOR = "<|end-of-read|>"
EOW = "<|end-of-write|>"

LATENCY_CHUNK_WORDS = {"low": 2, "medium": 4, "high": 8}

SIMT_INSTRUCTION = (
    "You are a professional simultaneous interpreter, your task is to translate "
    "the following {src_lang} text into {tgt_lang} with {latency} latency."
)
OMT_INSTRUCTION = (
    "You are a professional translator, your task is to translate the "
    "following text from {src_lang} into {tgt_lang}."
)

DIALECT_NAMES = {
    "EG": "Egyptian Arabic",
    "SA": "Saudi Arabic",
    "LB": "Lebanese Arabic",
    "MA": "Moroccan Arabic",
    "JO": "Jordanian Arabic",
    "PS": "Palestinian Arabic",
    "SY": "Syrian Arabic",
    "OM": "Omani Arabic",
    "YE": "Yemeni Arabic",
    "SD": "Sudanese Arabic",
    "LY": "Libyan Arabic",
    "MR": "Mauritanian Arabic",
    "TN": "Tunisian Arabic",
}


def _split_into_k_chunks(words, k):
    """Cuts a word list into k pieces of almost equal length and joins each piece back
    into a string."""
    n = len(words)
    return [" ".join(words[(i * n) // k:((i + 1) * n) // k]) for i in range(k)]


def chunk_sentence_pair(source, target, chunk_size):
    """Splits source and target into the same number K of chunks, where K comes from
    chunk_size applied to the longer side. Each chunk is a proportional fraction of its
    own side's word count, so chunk i still lines up when the two languages use different
    numbers of words."""
    source_words, target_words = source.split(), target.split()
    if not source_words or not target_words:
        return [], []
    # Round up the longer side to get K, then clamp so neither side gets empty chunks.
    k = max(1, -(-max(len(source_words), len(target_words)) // chunk_size))
    k = min(k, len(source_words), len(target_words))
    return _split_into_k_chunks(source_words, k), _split_into_k_chunks(target_words, k)


def whole_sentence_chunk(text):
    """Wraps a sentence as a single chunk, or gives an empty list if the text is blank."""
    text = text.strip()
    return [text] if text else []


def load_precomputed_chunks(path):
    """Loads generate_semantic_chunks.py output into a {(conv_id, turn_idx): entry} lookup."""
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    lookup = {}
    for entry in entries:
        lookup[(entry["conv_id"], entry["turn_idx"])] = entry
    return lookup


def align_chunks(source_chunks, target_chunks):
    """Forces both sides to have the same number of chunks by merging the extra chunks on
    the longer side into its last chunk. Interleaving needs the counts to match."""
    source_chunks, target_chunks = list(source_chunks), list(target_chunks)
    while len(source_chunks) > len(target_chunks) and len(source_chunks) > 1:
        last = source_chunks.pop()
        source_chunks[-1] = f"{source_chunks[-1]} {last}"
    while len(target_chunks) > len(source_chunks) and len(target_chunks) > 1:
        last = target_chunks.pop()
        target_chunks[-1] = f"{target_chunks[-1]} {last}"
    n = min(len(source_chunks), len(target_chunks))
    return source_chunks[:n], target_chunks[:n]


def interleave_chunks(source_chunks, target_chunks):
    """Renders already-aligned (equal-length) chunk lists into the interleaved
    <|end-of-read|> / <|end-of-write|> string."""
    if not source_chunks:
        return None
    parts = []
    for i, (sc, tc) in enumerate(zip(source_chunks, target_chunks)):
        sep = " " if i > 0 else ""
        parts.append(f"{sep}{sc}{EOR} {tc}{EOW}")
    return "".join(parts)


def build_simt_output(source, target, chunk_words=None, fixed_chunks=None):
    """Produces the interleaved training string for one sentence pair. Uses fixed_chunks when
    a segmentation is supplied, otherwise falls back to the word-count split."""
    if fixed_chunks is not None:
        source_chunks, target_chunks = fixed_chunks
        source_chunks, target_chunks = align_chunks(source_chunks, target_chunks)
    else:
        source_chunks, target_chunks = chunk_sentence_pair(source, target, chunk_words)
    return interleave_chunks(source_chunks, target_chunks)


def build_examples(source, target, src_lang, tgt_lang, granularity="chunk", precomputed_entry=None, stats=None):
    """Builds the 3 SiMT examples (one per latency) plus 1 offline example for one pair.

    With granularity="chunk", precomputed_entry supplies its own segmentation per latency
    if given, otherwise the word-count heuristic is used. granularity="turn" always uses one
    whole-sentence chunk, whatever precomputed_entry says."""
    examples = []

    if granularity == "chunk" and precomputed_entry is None and stats is not None and stats.get("_chunks_path_set"):
        # --chunks_path was given but this turn has no entry, so count the miss.
        stats["fallback_to_heuristic"] = stats.get("fallback_to_heuristic", 0) + 1

    # One SiMT example per latency level. Where the chunk boundaries come from depends on
    # the granularity and on whether this turn has a precomputed entry.
    for latency, chunk_words in LATENCY_CHUNK_WORDS.items():
        fixed_chunks = None
        if granularity == "turn":
            fixed_chunks = (whole_sentence_chunk(source), whole_sentence_chunk(target))
        elif precomputed_entry is not None:
            latency_entry = precomputed_entry["chunks"][latency]
            fixed_chunks = (latency_entry["source_chunks"], latency_entry["target_chunks"])

        output = build_simt_output(source, target, chunk_words=chunk_words, fixed_chunks=fixed_chunks)
        if output is None:
            continue
        examples.append(
            {
                "instruction": SIMT_INSTRUCTION.format(src_lang=src_lang, tgt_lang=tgt_lang, latency=latency),
                "input": "",
                "output": output,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "latency": latency,
            }
        )

    # The offline example has no chunking at all: full source in, full target out.
    examples.append(
        {
            "instruction": OMT_INSTRUCTION.format(src_lang=src_lang, tgt_lang=tgt_lang),
            "input": source,
            "output": target,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
        }
    )
    return examples


def _register_keep_flags(dialectal_texts, country, register_filter,
                          max_msa_markers, min_dialectness):
    """Keep/drop decision for the register-filter, applied to the Arabic side of each pair
    (the side that carries register whichever way we translate). Returns
    (keep_flags, scores_for_logging).

      - "lexicon": keep if count_msa_markers(...) <= max_msa_markers. Cheap but low recall.
      - "aldi":    keep if ALDi dialectness >= min_dialectness. Needs a model.
    """
    if register_filter == "lexicon":
        from dialect_lexicon import count_msa_markers
        scores = [count_msa_markers(t, country) for t in dialectal_texts]
        keep = [n <= max_msa_markers for n in scores]
        return keep, scores
    if register_filter == "aldi":
        from score_dialectness import ALDiScorer
        scores = ALDiScorer().score(dialectal_texts)  # batched internally
        keep = [s >= min_dialectness for s in scores]
        return keep, scores
    raise ValueError(f"unknown register_filter {register_filter!r}")


def build_dataset(country, split, direction, granularity="chunk", chunks_path=None,
                  domain=None, upweight_domain=None, domain_weight=1,
                  register_filter="none", max_msa_markers=0, min_dialectness=0.4,
                  register_min_words=0, register_action="drop", downweight_keep_frac=0.0, seed=0):
    """Turns one Alexandria dialect split into the full list of training examples.

    Walks every conversation turn, optionally filters by domain and by dialect register,
    then builds the SiMT and offline examples for the turns that survive."""
    # Step 1: load the split and, if asked, keep only one conversation domain.
    dataset = load_dataset("UBC-NLP/alexandria", name=country, split=split)
    if domain is not None:
        dataset = dataset.filter(lambda x: x["domain"] == domain)
    tgt_dialect_name = DIALECT_NAMES.get(country, f"Arabic ({country})")

    precomputed_lookup = load_precomputed_chunks(chunks_path) if chunks_path else None
    stats = {"_chunks_path_set": chunks_path is not None, "fallback_to_heuristic": 0, "total_turns": 0}

    # Pass 1: collect the pairs. Scoring is deferred so ALDi can batch the whole split.
    pairs = []
    for conv in dataset:
        conv_id = conv["conv_id"]
        # How many copies of this conversation's turns to emit later, for domain upweighting.
        repeat = domain_weight if (upweight_domain is not None and conv.get("domain") == upweight_domain) else 1
        for turn_idx, (eng_turn, dia_turn) in enumerate(zip(conv["english_conversation"], conv["dialectal_conversation"])):
            eng_text = eng_turn["text"].strip()
            dia_text = dia_turn["text"].strip()
            if not eng_text or not dia_text:
                continue

            # The direction decides which language is the source and which is the target.
            if direction == "en2ar":
                source, target, src_lang, tgt_lang = eng_text, dia_text, "English", tgt_dialect_name
            else:
                source, target, src_lang, tgt_lang = dia_text, eng_text, tgt_dialect_name, "English"

            precomputed_entry = precomputed_lookup.get((conv_id, turn_idx)) if precomputed_lookup else None
            stats["total_turns"] += 1
            pairs.append({
                "source": source, "target": target, "src_lang": src_lang, "tgt_lang": tgt_lang,
                "dialectal_text": dia_text, "precomputed_entry": precomputed_entry, "repeat": repeat,
            })

    # Step 2: register-filter scoring (on the Arabic side), then keep/drop or downweight.
    n_reg_dropped = 0
    if register_filter != "none":
        keep_flags, reg_scores = _register_keep_flags(
            [p["dialectal_text"] for p in pairs], country, register_filter,
            max_msa_markers, min_dialectness)
        rng = random.Random(seed)
        n_short_exempt = 0
        for p, keep, score in zip(pairs, keep_flags, reg_scores):
            p["_reg_score"] = score
            # Short turns score low on ALDi for being register-neutral, not MSA, so exempt them.
            if len(p["dialectal_text"].split()) < register_min_words:
                keep = True
                n_short_exempt += 1
            if keep:
                p["_reg_keep"] = True
            elif register_action == "downweight":
                # deterministic per-pair coin so re-runs are reproducible
                p["_reg_keep"] = rng.random() < downweight_keep_frac
            else:  # drop
                p["_reg_keep"] = False
            if not p["_reg_keep"]:
                n_reg_dropped += 1
        # Report how much the filter removed, so the effect on data size is visible in the log.
        gate_note = f" ({n_short_exempt} short turns < {register_min_words} words exempted)" if register_min_words else ""
        if register_filter == "aldi":
            kept_scores = [p["_reg_score"] for p in pairs if p["_reg_keep"]]
            mean_kept = sum(kept_scores) / len(kept_scores) if kept_scores else 0.0
            print(f"Register-filter (aldi >= {min_dialectness}{gate_note}): dropped {n_reg_dropped}/{len(pairs)} "
                  f"pairs; mean ALDi of kept = {mean_kept:.3f}", file=sys.stderr)
        else:
            print(f"Register-filter (lexicon, <= {max_msa_markers} MSA markers{gate_note}): "
                  f"dropped {n_reg_dropped}/{len(pairs)} pairs", file=sys.stderr)
    else:
        for p in pairs:
            p["_reg_keep"] = True

    # Pass 2: build examples for kept pairs, repeated for domain upweighting.
    all_examples = []
    n_upweighted = 0
    for p in pairs:
        if not p["_reg_keep"]:
            continue
        exs = build_examples(p["source"], p["target"], p["src_lang"], p["tgt_lang"],
                             granularity=granularity, precomputed_entry=p["precomputed_entry"], stats=stats)
        all_examples.extend(exs)
        # repeat > 1 only for the upweighted domain: append the same examples again.
        for _ in range(p["repeat"] - 1):
            all_examples.extend(exs)
            n_upweighted += len(exs)

    # Step 3: report how well the precomputed chunks covered the data and how much was duplicated.
    if chunks_path:
        n_fallback = stats["fallback_to_heuristic"]
        n_total = stats["total_turns"]
        print(
            f"Precomputed chunks: {n_total - n_fallback}/{n_total} turns matched "
            f"({n_fallback} fell back to the word-count heuristic)",
            file=sys.stderr,
        )
    if upweight_domain is not None:
        print(f"Domain upweight ({upweight_domain!r} x{domain_weight}): "
              f"added {n_upweighted} duplicated examples", file=sys.stderr)

    return all_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="EG", help="Alexandria country/dialect code")
    parser.add_argument("--split", default="train")
    parser.add_argument("--direction", choices=["en2ar", "ar2en"], default="en2ar")
    parser.add_argument("--granularity", choices=["chunk", "turn"], default="chunk",
                         help="'chunk' = a different chunking per latency; 'turn' = one "
                              "whole-sentence chunk for every latency")
    parser.add_argument("--chunks_path", default=None,
                         help="Precomputed LLM chunks; turns with no entry fall back to the "
                              "word-count heuristic. Only used with --granularity chunk.")
    # --- domain-match (keep/upweight a domain) ---
    parser.add_argument("--domain", default=None,
                         help="Keep ONLY this conv['domain'] (e.g. 'Commerce and transactions')")
    parser.add_argument("--upweight_domain", default=None,
                         help="Keep all domains, but duplicate this one --domain_weight times")
    parser.add_argument("--domain_weight", type=int, default=1,
                         help="Duplication factor for --upweight_domain (2 = one extra copy).")
    # --- register-filter (drop/downweight MSA-leaning targets) ---
    parser.add_argument("--register_filter", choices=["none", "lexicon", "aldi"], default="none",
                         help="'lexicon' = cheap MSA-marker count (EG/LB only); 'aldi' = "
                              "Sentence-ALDi dialectness score (downloads a model)")
    parser.add_argument("--max_msa_markers", type=int, default=0,
                         help="lexicon filter: drop a pair with more MSA markers than this")
    parser.add_argument("--min_dialectness", type=float, default=0.4,
                         help="aldi filter: drop a pair scoring below this (0=MSA, 1=dialect)")
    parser.add_argument("--register_min_words", type=int, default=0,
                         help="Exempt Arabic turns shorter than this from the register-filter; "
                              "about 6 works well on Alexandria")
    parser.add_argument("--register_action", choices=["drop", "downweight"], default="drop",
                         help="what to do with pairs that fail the register threshold.")
    parser.add_argument("--downweight_keep_frac", type=float, default=0.0,
                         help="with --register_action downweight: fraction of failing pairs to keep (0..1).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible downweight sampling.")
    parser.add_argument("--output", default="data/mt_data/train_data/Arabic-EG-SiMT-OMT.json")
    args = parser.parse_args()

    examples = build_dataset(args.country, args.split, args.direction,
                              granularity=args.granularity, chunks_path=args.chunks_path,
                              domain=args.domain, upweight_domain=args.upweight_domain,
                              domain_weight=args.domain_weight, register_filter=args.register_filter,
                              max_msa_markers=args.max_msa_markers, min_dialectness=args.min_dialectness,
                              register_min_words=args.register_min_words, register_action=args.register_action,
                              downweight_keep_frac=args.downweight_keep_frac, seed=args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    # SiMT rows are the ones carrying a "latency" key; everything else is an offline row.
    n_simt = sum(1 for ex in examples if "latency" in ex)
    n_omt = len(examples) - n_simt
    print(f"Wrote {len(examples)} examples ({n_simt} SiMT, {n_omt} OMT) to {args.output}")
