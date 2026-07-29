"""
Same chunk-alignment check as analyze_chunk_alignment.py, but split by where
the chunks came from: the LLM chunker or the word-count fallback.

Input: a chunks-*.json file (the derived training file does not record the source).
Output: matched vs shuffled similarity per source and per latency level.
"""
import argparse
import json
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

from sentence_transformers import SentenceTransformer, util

LATENCIES = ["low", "medium", "high"]


def matched_vs_shuffled(source_chunks, target_chunks, model, rng):
    """Average cosine similarity of the real chunk pairing, and of a shuffled control pairing.

    The shuffle is a derangement, so no chunk is left paired with its own translation. A real
    alignment should score clearly higher than the shuffled one."""
    src_emb = model.encode(source_chunks, convert_to_tensor=True, show_progress_bar=False)
    tgt_emb = model.encode(target_chunks, convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(src_emb, tgt_emb).numpy()
    n = len(source_chunks)
    matched = [sims[i][i] for i in range(n)]
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)) or n < 2:
            break
    shuffled = [sims[i][perm[i]] for i in range(n)]
    return sum(matched) / n, sum(shuffled) / n


def main():
    """Run the alignment check separately for LLM-produced chunks and fallback chunks.

    This tells us whether the LLM chunker is actually better aligned than the mechanical
    word-count backup it falls back to."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_path", required=True)
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # One seeded RNG shared by every shuffle, so the control pairing is reproducible.
    rng = random.Random(args.seed)

    # The chunks file records, for every entry and every latency, the chunks themselves plus a flag
    # saying whether that latency fell back to the word-count backup.
    with open(args.chunks_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Loading {args.model} (first run downloads the model)...")
    model = SentenceTransformer(args.model)

    # Sort every usable chunking into one of six buckets: (llm or fallback) x (low/medium/high).
    # Entries with fewer than two chunks, or with mismatched counts, cannot be checked.
    groups = {(src, lat): [] for src in ("llm", "fallback") for lat in LATENCIES}

    for entry in entries:
        fallback_by_latency = entry.get("fallback_by_latency", {})
        for lat in LATENCIES:
            chunk_entry = entry.get("chunks", {}).get(lat)
            if not chunk_entry:
                continue
            source_chunks = chunk_entry["source_chunks"]
            target_chunks = chunk_entry["target_chunks"]
            # A single chunk cannot be deranged, and unequal counts have no diagonal to read.
            if len(source_chunks) < 2 or len(source_chunks) != len(target_chunks):
                continue
            # The fallback flag is what decides which of the two source buckets this chunking is in.
            src = "fallback" if fallback_by_latency.get(lat) else "llm"
            groups[(src, lat)].append((source_chunks, target_chunks))

    # Average each bucket and report it, plus how often matched beat shuffled.
    for src in ("llm", "fallback"):
        print("=" * 70)
        print(f"SOURCE = {src}")
        print("=" * 70)
        for lat in LATENCIES:
            pairs_list = groups[(src, lat)]
            if not pairs_list:
                print(f"  {lat}: no usable (>=2 chunks) examples in this group")
                continue
            # Score every chunking in this bucket, then average the two columns.
            matched_scores, shuffled_scores = [], []
            for source_chunks, target_chunks in pairs_list:
                m, s = matched_vs_shuffled(source_chunks, target_chunks, model, rng)
                matched_scores.append(m)
                shuffled_scores.append(s)
            # matched-wins counts the chunkings where the real pairing beat its shuffled control.
            n_total = len(pairs_list)
            n_wins = sum(1 for m, s in zip(matched_scores, shuffled_scores) if m > s)
            avg_matched = sum(matched_scores) / n_total
            avg_shuffled = sum(shuffled_scores) / n_total
            print(f"  {lat}: n={n_total:4d}  matched={avg_matched:.4f}  shuffled={avg_shuffled:.4f}  "
                  f"matched-wins={n_wins}/{n_total} ({100 * n_wins / n_total:.1f}%)")
        print()


if __name__ == "__main__":
    main()
