"""
Checks whether chunk i of the source really matches chunk i of the target in
the SiMT training data.

Input: a training JSON file with interleaved chunk markers in each "output".
Output: printed summary comparing matched similarity against a shuffled control.
Needs to download a multilingual sentence encoder on first run.
"""
import argparse
import json
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

from sentence_transformers import SentenceTransformer, util

EOR = "<|end-of-read|>"
EOW = "<|end-of-write|>"
PAIR_RE = re.compile(re.escape(EOR) + r"\s*(.*?)" + re.escape(EOW), re.DOTALL)


def parse_chunks(output):
    """Recover (source_chunk, target_chunk) pairs from an interleaved output string."""
    # Layout is: sc1<eor> tc1<eow> sc2<eor> tc2<eow> ...
    matches = list(PAIR_RE.finditer(output))
    pairs = []
    prev_end = 0
    for m in matches:
        source_chunk = output[prev_end:m.start()].strip()
        target_chunk = m.group(1).strip()
        pairs.append((source_chunk, target_chunk))
        prev_end = m.end()
    return pairs


def main():
    """Measure how well chunk i of the source lines up with chunk i of the target.

    Each chunk pair is embedded with a multilingual sentence encoder and compared by cosine
    similarity. The same chunks are then compared in a shuffled order as a control: if the real
    pairing is meaningful, matched similarity should clearly beat shuffled similarity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mt_data/train_data/Arabic-EG-SiMT-OMT.json")
    parser.add_argument("--latency", default=None, choices=["low", "medium", "high", None])
    parser.add_argument("--sample_size", type=int, default=400)
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show_worst", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)

    # Step 1: keep only the simultaneous rows (the offline rows have no latency field and no
    # chunk markers), then split each one back into chunk pairs. A row with a single chunk
    # cannot be shuffled, so at least two are required.
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    simt_rows = [r for r in data if "latency" in r]
    if args.latency:
        simt_rows = [r for r in simt_rows if r["latency"] == args.latency]

    parsed = []
    for row in simt_rows:
        pairs = parse_chunks(row["output"])
        if len(pairs) >= 2:
            parsed.append((row["latency"], pairs))

    print(f"Loaded {len(simt_rows)} SiMT rows; {len(parsed)} have >=2 chunks (usable for this check)")
    # Embedding every row would be slow, so only --sample_size of them are actually checked.
    if len(parsed) > args.sample_size:
        parsed = random.sample(parsed, args.sample_size)
    print(f"Sampling {len(parsed)} examples for embedding-based alignment check\n")

    print(f"Loading {args.model} (first run downloads the model)...")
    model = SentenceTransformer(args.model)

    # Step 2: for each example, embed its source and target chunks and build the full similarity
    # matrix, then compare the diagonal (the real pairing) with a shuffled pairing.
    per_example_results = []
    for latency, pairs in parsed:
        sources = [p[0] for p in pairs]
        targets = [p[1] for p in pairs]
        src_emb = model.encode(sources, convert_to_tensor=True, show_progress_bar=False)
        tgt_emb = model.encode(targets, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(src_emb, tgt_emb).numpy()  # sims[i][j] = sim(src_i, tgt_j)

        n = len(pairs)
        # The diagonal is the real pairing: source chunk i against target chunk i.
        matched = [sims[i][i] for i in range(n)]
        # Derangement: a shuffle with no chunk left in its own position. Retry until every position
        # moved, so the control never accidentally contains a correct pair.
        perm = list(range(n))
        while True:
            random.shuffle(perm)
            if all(perm[i] != i for i in range(n)) or n < 2:
                break
        shuffled = [sims[i][perm[i]] for i in range(n)]

        # One record per example: both means and the gap between them, plus the chunks themselves
        # so the worst cases can be printed at the end.
        matched_mean = sum(matched) / n
        shuffled_mean = sum(shuffled) / n
        per_example_results.append({
            "latency": latency,
            "n_chunks": n,
            "matched_mean": matched_mean,
            "shuffled_mean": shuffled_mean,
            "gap": matched_mean - shuffled_mean,
            "pairs": pairs,
        })

    # Step 3: sort by gap (matched minus shuffled), so the worst examples come first and can be
    # printed at the end for manual inspection.
    per_example_results.sort(key=lambda r: r["gap"])

    # Whole-sample view: the two averages, and how many examples the real pairing actually won.
    n_total = len(per_example_results)
    n_matched_wins = sum(1 for r in per_example_results if r["gap"] > 0)
    avg_matched = sum(r["matched_mean"] for r in per_example_results) / n_total
    avg_shuffled = sum(r["shuffled_mean"] for r in per_example_results) / n_total

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Examples checked: {n_total}")
    print(f"Mean matched (chunk i vs chunk i) similarity:   {avg_matched:.4f}")
    print(f"Mean shuffled (chunk i vs random chunk j) sim:  {avg_shuffled:.4f}")
    print(f"Matched beats shuffled in {n_matched_wins}/{n_total} examples ({100*n_matched_wins/n_total:.1f}%)")
    # Step 4: the same numbers per latency level. Finer chunking (low latency) means shorter
    # chunks, which are harder to align.
    print()
    print("By latency level:")
    for lat in ["low", "medium", "high"]:
        subset = [r for r in per_example_results if r["latency"] == lat]
        if not subset:
            continue
        m = sum(r["matched_mean"] for r in subset) / len(subset)
        s = sum(r["shuffled_mean"] for r in subset) / len(subset)
        wins = sum(1 for r in subset if r["gap"] > 0)
        print(f"  {lat:8s} n={len(subset):4d}  matched={m:.4f}  shuffled={s:.4f}  matched-wins={wins}/{len(subset)} ({100*wins/len(subset):.1f}%)")

    print()
    print("=" * 70)
    print(f"WORST {args.show_worst} EXAMPLES (matched - shuffled most negative, i.e. shuffled looked MORE aligned)")
    print("=" * 70)
    # Step 5: the list is already sorted by gap, so the front of it is the worst cases. Printing
    # the chunks lets a human check whether the misalignment is real.
    for r in per_example_results[:args.show_worst]:
        print(f"\n[{r['latency']}] n_chunks={r['n_chunks']}  matched={r['matched_mean']:.4f}  shuffled={r['shuffled_mean']:.4f}  gap={r['gap']:+.4f}")
        for i, (sc, tc) in enumerate(r["pairs"]):
            print(f"  chunk {i}: EN={sc!r}  AR={tc!r}")


if __name__ == "__main__":
    main()
