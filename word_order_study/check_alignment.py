"""Measure whether chunk i of the English really matches chunk i of the target.

Reads the chunks files under data/, embeds each chunk pair with a multilingual sentence encoder,
and reports the mean cosine similarity of matched pairs against shuffled pairs as a control.
Model chunks and word-count fallback chunks are reported separately. Writes alignment_check.md.

Usage:
  python word_order_study/check_alignment.py --languages vietnamese korean --methods generic
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LATENCIES = ["low", "medium", "high"]


def collect_pairs(entries, latency):
    """Split this latency's chunk pairs into the ones the model produced and the fallback ones."""
    llm, fallback = [], []
    for e in entries:
        c = e.get("chunks", {}).get(latency)
        if not c:
            continue
        pairs = list(zip(c["source_chunks"], c["target_chunks"]))
        if e["fallback_by_latency"].get(latency):
            fallback.extend(pairs)
        else:
            llm.extend(pairs)
    return llm, fallback


def score(model, pairs, sample_size, rng):
    """Return (matched, shuffled, gap, n) similarity for a list of chunk pairs, or None if too few."""
    from sentence_transformers import util

    if len(pairs) < 20:
        return None
    if len(pairs) > sample_size:
        pairs = rng.sample(pairs, sample_size)
    src = [p[0] for p in pairs]
    tgt = [p[1] for p in pairs]

    emb_src = model.encode(src, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
    emb_tgt = model.encode(tgt, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)

    matched = float(util.pairwise_cos_sim(emb_src, emb_tgt).mean())

    # Control: rotate the targets by one, so no chunk keeps its real partner.
    order = list(range(1, len(pairs))) + [0]
    shuffled = float(util.pairwise_cos_sim(emb_src, emb_tgt[order]).mean())

    return matched, shuffled, matched - shuffled, len(pairs)


def main():
    """Score every (language, method, latency) and print one table."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+",
                    default=["vietnamese", "msa", "korean", "saudi", "egyptian"])
    ap.add_argument("--methods", nargs="+", default=["generic", "specific"])
    ap.add_argument("--sample_size", type=int, default=600, help="chunk pairs scored per cell")
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=os.path.join(DATA, "alignment_check.md"))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    from sentence_transformers import SentenceTransformer
    print(f"Loading encoder: {args.model}")
    model = SentenceTransformer(args.model)

    rows = []
    for lang in args.languages:
        for method in args.methods:
            path = os.path.join(DATA, lang, f"chunks-{method}.json")
            if not os.path.exists(path):
                print(f"  skip (missing): {path}")
                continue
            with open(path, encoding="utf-8") as f:
                entries = json.load(f)
            for latency in LATENCIES:
                llm, fallback = collect_pairs(entries, latency)
                for source_label, pairs in (("llm", llm), ("fallback", fallback)):
                    s = score(model, pairs, args.sample_size, rng)
                    if s is None:
                        continue
                    matched, shuffled, gap, n = s
                    rows.append((lang, method, latency, source_label, matched, shuffled, gap, n))
                    print(f"  {lang:11} {method:8} {latency:6} {source_label:8} "
                          f"matched={matched:.3f} shuffled={shuffled:.3f} gap={gap:+.3f} (n={n})")

    lines = ["# Chunk alignment check", "",
             "`matched` = mean similarity of English chunk i to target chunk i.",
             "`shuffled` = the same chunks paired with someone else's partner (control).",
             "`gap` = matched - shuffled. A large gap means chunk i really does correspond to chunk i.",
             "A gap near zero means the chunks are misaligned, even if they rebuild the sentence.", "",
             "| Language | Method | Latency | Chunks from | matched | shuffled | gap | n |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for lang, method, latency, source_label, matched, shuffled, gap, n in rows:
        lines.append(f"| {lang} | {method} | {latency} | {source_label} | "
                     f"{matched:.3f} | {shuffled:.3f} | {gap:+.3f} | {n} |")

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
