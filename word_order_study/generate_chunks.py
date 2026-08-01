"""Chunk generator for the word-order study.

Reads the Step 1 parallel files and writes a chunks file for build_simt_sft.py.
Two backends: heuristic (word-count split, no model) and llama (a prompted chat model,
checked by chunk_validate.py, falling back to the word-count split when it fails).

Usage:
  python word_order_study/generate_chunks.py --language korean --method generic \
      --backend llama --output word_order_study/data/korean/chunks-generic.json
"""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)                                # chunk_prompts
sys.path.insert(0, os.path.join(REPO, "east_scripts", "data"))  # reuse the Arabic pipeline's core

import chunk_prompts
from build_arabic_simt_sft_data import chunk_sentence_pair, LATENCY_CHUNK_WORDS

EXT = {"vietnamese": "vi", "msa": "ar", "korean": "ko", "saudi": "ar", "egyptian": "ar"}


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def heuristic_entry(source, target):
    """Split both sentences by word count. Nothing can fail, so no latency is a fallback."""
    chunks = {lat: chunk_sentence_pair(source, target, w) for lat, w in LATENCY_CHUNK_WORDS.items()}
    return chunks, {lat: False for lat in LATENCY_CHUNK_WORDS}, {lat: None for lat in LATENCY_CHUNK_WORDS}


def retry_fallbacks(args, src, tgt):
    """Second pass over an existing chunks file: retry only the latencies that failed."""
    import torch
    from generate_semantic_chunks import ChunkGenerator
    from chunk_validate import parse_single_latency

    # Step 1: reopen the chunks file written by the first pass.
    with open(args.output, encoding="utf-8") as f:
        entries = json.load(f)

    def count_fb():
        """Count how many sentences still fail, per latency."""
        c = Counter()
        for e in entries:
            for lat in LATENCY_CHUNK_WORDS:
                if e["fallback_by_latency"].get(lat):
                    c[lat] += 1
        return c

    before = count_fb()
    total = len(entries)
    n_fb_entries = sum(1 for e in entries if e.get("fallback"))
    print(f"Loading chunker model: {args.model}")
    cg = ChunkGenerator(args.model, min_similarity=args.min_reconstruction_similarity)

    # Step 2: ask the model for one latency only. Sampling is on, so each retry can differ.
    @torch.no_grad()
    def sample_single(source, target, method, language, latency):
        prompt = chunk_prompts.build_prompt_single(method, language, source, target, latency)
        pt = cg.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = cg.tokenizer(pt, add_special_tokens=False, return_tensors="pt").to(cg.model.device)
        out = cg.model.generate(
            **inputs, max_new_tokens=args.retry_max_new_tokens, do_sample=True,
            temperature=args.retry_temperature, top_p=0.95, num_beams=1,
            eos_token_id=cg.eos_token_ids,
            pad_token_id=cg.tokenizer.pad_token_id or cg.tokenizer.eos_token_id)
        return cg.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    out_path = args.retry_output or args.output
    budget = args.max_examples  # cap on fallback entries to retry; None = all
    # Step 3: go through the failed sentences only, retry each failed latency, and save every 100.
    recovered, processed = Counter(), 0
    for e in entries:
        if not e.get("fallback"):
            continue
        if budget is not None and processed >= budget:
            break
        idx = e["idx"]
        s, t = src[idx].strip(), tgt[idx].strip()
        method, language = e["method"], e["language"]
        for lat in LATENCY_CHUNK_WORDS:
            if not e["fallback_by_latency"].get(lat):
                continue
            for _ in range(args.retry_samples):
                try:
                    aligned = parse_single_latency(
                        sample_single(s, t, method, language, lat), s, t, args.min_reconstruction_similarity)
                except Exception:
                    aligned = None
                if aligned is not None:
                    sc, tc = aligned
                    e["chunks"][lat] = {"source_chunks": sc, "target_chunks": tc}
                    e["fallback_by_latency"][lat] = False
                    e["fallback_reason_by_latency"][lat] = None
                    recovered[lat] += 1
                    break
        e["fallback"] = any(e["fallback_by_latency"].values())
        processed += 1
        if processed % 100 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            print(f"  ...retried {processed}/{n_fb_entries} fallback entries", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    after = count_fb()
    print(f"\nRETRY {args.language}/{args.method}: retried {processed} of {n_fb_entries} fallback entries "
          f"(samples<={args.retry_samples}, T={args.retry_temperature}) -> {out_path}")
    for lat in LATENCY_CHUNK_WORDS:
        print(f"  {lat}: {before[lat]}/{total} ({100*before[lat]/max(1,total):.1f}%) -> "
              f"{after[lat]}/{total} ({100*after[lat]/max(1,total):.1f}%)   recovered {recovered[lat]}")


def main():
    """Chunk one (language, method) pair at three latencies and write the chunks file."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=chunk_prompts.LANGUAGES)
    ap.add_argument("--method", required=True, choices=["generic", "specific"])
    ap.add_argument("--backend", required=True, choices=["heuristic", "llama"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help="llama backend only; any HF causal-LM chat model")
    ap.add_argument("--min_reconstruction_similarity", type=float, default=0.97)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip idx already in --output that did NOT fall back; retry fallbacks (safe "
                         "against a walltime timeout - re-submit the same job to continue)")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--retry_fallbacks", action="store_true",
                    help="SECOND PASS: re-chunk only the fallback (idx, latency) pairs in --output using a "
                         "single-latency prompt + sampling + best-of-N, then merge and rewrite. Cuts the "
                         "fallback rate without touching the already-good chunks.")
    ap.add_argument("--retry_samples", type=int, default=3, help="best-of-N samples per failed latency")
    ap.add_argument("--retry_temperature", type=float, default=0.7)
    ap.add_argument("--retry_max_new_tokens", type=int, default=768)
    ap.add_argument("--retry_output", default=None,
                    help="where to write retried chunks (default: overwrite --output)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Step 1: load the parallel sentences. Line i of one file matches line i of the other.
    ext = EXT[args.language]
    src = read_lines(os.path.join(DATA, args.language, f"{args.split}.en"))
    tgt = read_lines(os.path.join(DATA, args.language, f"{args.split}.{ext}"))
    if len(src) != len(tgt):
        raise SystemExit(f"line counts differ: {len(src)} en vs {len(tgt)} {ext}")
    n = len(src) if args.max_examples is None else min(args.max_examples, len(src))
    tgt_lang = chunk_prompts.TGT_LANG_NAME[args.language]

    # Step 2: retry mode is a separate job, it only re-does what failed in an earlier run.
    if args.retry_fallbacks:
        if not os.path.exists(args.output):
            raise SystemExit(f"--retry_fallbacks needs an existing --output; {args.output} not found")
        retry_fallbacks(args, src, tgt)
        return

    # Step 3: set up the chunker. The heuristic backend leaves gen as None and needs no torch.
    gen = None
    if args.backend == "llama":
        import torch
        from generate_semantic_chunks import ChunkGenerator
        from chunk_validate import parse_chunk_response_wo as parse_chunk_response

        class _Gen:
            """Sends one sentence pair to the model and returns the validated chunks."""

            def __init__(self, model):
                self.cg = ChunkGenerator(model, min_similarity=args.min_reconstruction_similarity)

            @torch.no_grad()
            def __call__(self, source, target):
                prompt = chunk_prompts.build_prompt(args.method, args.language, source, target)
                pt = self.cg.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                inputs = self.cg.tokenizer(pt, add_special_tokens=False, return_tensors="pt").to(self.cg.model.device)
                out = self.cg.model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False, num_beams=1,
                    eos_token_id=self.cg.eos_token_ids,
                    pad_token_id=self.cg.tokenizer.pad_token_id or self.cg.tokenizer.eos_token_id)
                resp = self.cg.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                return parse_chunk_response(resp, source, target, args.min_reconstruction_similarity)

        print(f"Loading chunker model: {args.model}")
        gen = _Gen(args.model)

    # Step 4: with --resume, keep the sentences that already chunked cleanly and redo the failures.
    entries, done = [], set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            prev = json.load(f)
        entries = [e for e in prev if not e.get("fallback")]
        done = {e["idx"] for e in entries}
        print(f"Resuming: {len(entries)} good entries kept, "
              f"{len(prev) - len(entries)} fallbacks will be retried", file=sys.stderr)

    # Step 5: chunk the sentences one pair at a time. The llama backend saves every save_every entries.
    for i in range(n):
        if i in done:
            continue
        s, t = src[i].strip(), tgt[i].strip()
        if not s or not t:
            continue
        if gen is None:
            chunks, fbl, rbl = heuristic_entry(s, t)
        else:
            try:
                chunks, fbl, rbl = gen(s, t)
            except Exception as e:
                chunks, fbl, rbl = heuristic_entry(s, t)
                rbl = {lat: f"generation_exception: {e}" for lat in LATENCY_CHUNK_WORDS}
                fbl = {lat: True for lat in LATENCY_CHUNK_WORDS}
        entries.append({
            "idx": i, "method": args.method, "language": args.language,
            "chunks": {lat: {"source_chunks": sc, "target_chunks": tc} for lat, (sc, tc) in chunks.items()},
            "fallback": any(fbl.values()), "fallback_by_latency": fbl, "fallback_reason_by_latency": rbl,
        })
        if gen is not None and len(entries) % args.save_every == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            print(f"  ...{i + 1}/{n} processed, {len(entries)} entries so far", file=sys.stderr)

    # Step 6: write the finished chunks file.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    # Step 7: report how often the word-count split was used instead of the model.
    n_fallback = sum(1 for e in entries if e["fallback"])
    fb_by_lat, reasons = Counter(), Counter()
    for e in entries:
        for lat, r in e["fallback_reason_by_latency"].items():
            if r is not None:
                fb_by_lat[lat] += 1
                reasons[f"{lat}:{r}"] += 1

    print(f"\n{args.language}/{args.method}/{args.backend}: {len(entries)} entries in {args.output}")
    print(f"  turns with >=1 fallback latency: {n_fallback}/{len(entries)} "
          f"({100 * n_fallback / max(1, len(entries)):.1f}%)")
    for lat in LATENCY_CHUNK_WORDS:
        print(f"  {lat}: {fb_by_lat[lat]}/{len(entries)} fell back to heuristic "
              f"({100 * fb_by_lat[lat] / max(1, len(entries)):.1f}%)")
    if reasons:
        print("  fallback reasons:", dict(reasons.most_common()))


if __name__ == "__main__":
    main()
