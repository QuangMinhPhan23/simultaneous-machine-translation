"""
DPO Stage 3 pairs: the human reference (chosen) vs. the Stage-2 checkpoint's own output
(rejected), a harder contrast than Stage 1.

Run generate_candidates.py against the Stage-2 DPO checkpoint first, so the negatives are
fresh mistakes. Pairs where the model is already close to the reference are dropped by
chrF++, since they carry little training signal.
"""
import argparse
import json
import os

import sacrebleu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_path", required=True,
                         help="Output of generate_candidates.py run against the Stage-2 checkpoint")
    parser.add_argument("--output", default="data/mt_data/dpo_data/stage3_pairs.json")
    parser.add_argument("--max_chrf", type=float, default=75.0,
                         help="Skip pairs whose prediction already scores at or above this "
                              "chrF++ against the reference")
    args = parser.parse_args()

    # Step 1: read the Stage-2 checkpoint's outputs from generate_candidates.py.
    with open(args.candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    # Step 2: pair each human reference against the model's own output for that prompt.
    pairs = []
    n_dropped = 0
    for c in candidates:
        chosen = c["reference"].strip()
        rejected = c["prediction"].strip()
        # An empty or identical prediction gives no contrast to learn from.
        if not rejected or rejected == chosen:
            n_dropped += 1
            continue
        # Measure how close the two sides are, and drop the pair if they nearly agree.
        score = sacrebleu.sentence_chrf(rejected, [chosen], word_order=2).score
        if score >= args.max_chrf:
            n_dropped += 1
            continue
        pairs.append({
            "prompt": f"{c['instruction']}\n{c['input']}" if c.get("input") else c["instruction"],
            "chosen": chosen,
            "rejected": rejected,
            "rejected_chrf": score,
        })

    # Step 3: write the pairs out, ready for DPO training.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(pairs)} Stage 3 pairs to {args.output} ({n_dropped} dropped as too-close/degenerate)")
