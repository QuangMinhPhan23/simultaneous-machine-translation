"""Step 3: build EAST interleaved SiMT + OMT training data for the word-order study.

Takes the Step 1 parallel files (data/<language>/<split>.en / .<ext>) plus a chunks file from
generate_chunks.py, and writes alpaca {"instruction","input","output"} JSON (what
finetune_lora_stage2.py expects) - one file per (language, method), ~4 examples per pair (3 SiMT
latencies + 1 offline). Any (idx, latency) missing from the chunks file falls back to a word-count
split (logged). --merge_min_words merges tiny chunks into a neighbour (default 1 = off). Interleave and
align helpers come from east_scripts/data/build_arabic_simt_sft_data.py so the token format matches."""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "east_scripts", "data"))

import chunk_prompts
from build_arabic_simt_sft_data import (
    EOR, EOW, LATENCY_CHUNK_WORDS, SIMT_INSTRUCTION, OMT_INSTRUCTION,
    align_chunks, chunk_sentence_pair, interleave_chunks, whole_sentence_chunk,
)

EXT = {"vietnamese": "vi", "msa": "ar", "korean": "ko", "saudi": "ar", "egyptian": "ar"}


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def merge_small(src_chunks, tgt_chunks, min_words):
    """Merge any aligned pair whose shorter side has < min_words words into the previous pair (or next
    if it is the first). Preserves equal counts and verbatim reconstruction."""
    if min_words <= 1 or len(src_chunks) <= 1:
        return src_chunks, tgt_chunks
    s, t = list(src_chunks), list(tgt_chunks)
    i = 0
    while i < len(s) and len(s) > 1:
        if min(len(s[i].split()), len(t[i].split())) < min_words:
            j = i - 1 if i > 0 else i + 1
            lo, hi = sorted((i, j))
            s[lo] = f"{s[lo]} {s[hi]}".strip()
            t[lo] = f"{t[lo]} {t[hi]}".strip()
            del s[hi]; del t[hi]
            i = max(0, lo)
        else:
            i += 1
    return s, t


def build(args):
    ext = EXT[args.language]
    src = read_lines(os.path.join(DATA, args.language, f"{args.split}.en"))
    tgt = read_lines(os.path.join(DATA, args.language, f"{args.split}.{ext}"))
    if len(src) != len(tgt):
        raise SystemExit(f"line counts differ: {len(src)} vs {len(tgt)}")

    precomputed = None
    if args.chunks_path:
        with open(args.chunks_path, encoding="utf-8") as f:
            precomputed = {e["idx"]: e for e in json.load(f)}

    src_lang, tgt_lang = "English", chunk_prompts.TGT_LANG_NAME[args.language]
    examples, stats = [], Counter()
    for idx, (s, t) in enumerate(zip(src, tgt)):
        if args.max_examples is not None and stats["pairs"] >= args.max_examples:
            break
        s, t = s.strip(), t.strip()
        if not s or not t:
            continue
        entry = precomputed.get(idx) if precomputed else None
        stats["pairs"] += 1

        for latency, words in LATENCY_CHUNK_WORDS.items():
            if args.granularity == "turn":
                sc, tc = whole_sentence_chunk(s), whole_sentence_chunk(t)
            elif entry is not None and latency in entry.get("chunks", {}):
                le = entry["chunks"][latency]
                sc, tc = align_chunks(le["source_chunks"], le["target_chunks"])
            else:
                if precomputed is not None:
                    stats[f"fallback_{latency}"] += 1
                sc, tc = chunk_sentence_pair(s, t, words)
            sc, tc = merge_small(sc, tc, args.merge_min_words)
            output = interleave_chunks(sc, tc)
            if output is None:
                continue
            examples.append({
                "instruction": SIMT_INSTRUCTION.format(src_lang=src_lang, tgt_lang=tgt_lang, latency=latency),
                "input": "", "output": output,
                "src_lang": src_lang, "tgt_lang": tgt_lang, "latency": latency,
            })

        examples.append({
            "instruction": OMT_INSTRUCTION.format(src_lang=src_lang, tgt_lang=tgt_lang),
            "input": s, "output": t, "src_lang": src_lang, "tgt_lang": tgt_lang,
        })

    return examples, stats


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=chunk_prompts.LANGUAGES)
    ap.add_argument("--method", default="generic", choices=["generic", "specific"],
                    help="label only (chunking comes from --chunks_path); used in the default output name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--chunks_path", default=None)
    ap.add_argument("--granularity", choices=["chunk", "turn"], default="chunk")
    ap.add_argument("--merge_min_words", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=None, help="cap pairs processed (smoke/debug)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join(DATA, args.language, f"simt-{args.method}.{args.split}.json")

    examples, stats = build(args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    n_simt = sum(1 for e in examples if "latency" in e)
    print(f"{args.language}/{args.method}: wrote {len(examples)} examples "
          f"({n_simt} SiMT, {len(examples) - n_simt} OMT) from {stats['pairs']} pairs -> {args.output}")
    if args.chunks_path:
        for lat in LATENCY_CHUNK_WORDS:
            print(f"  {lat}: {stats[f'fallback_{lat}']}/{stats['pairs']} pairs fell back to heuristic "
                  f"(idx/latency missing from chunks file)")


if __name__ == "__main__":
    main()
