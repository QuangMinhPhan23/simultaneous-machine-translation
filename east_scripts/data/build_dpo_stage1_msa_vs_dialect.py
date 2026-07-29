"""
DPO Stage 1 pairs: the human dialectal reference (chosen) vs. the SFT checkpoint's own
output (rejected). This is the easy contrast, before the harder later stages.

Input is a generate_candidates.py file made with SFT_MODEL; output is prompt/chosen/rejected
JSON. --max_chrf drops near-duplicate pairs, which are a known cause of unstable DPO
training.
"""
import argparse
import json
import os

import sacrebleu


def build_prompt(instruction, input_text):
    """Joins the instruction and the source text into the single prompt string DPO expects."""
    return f"{instruction}\n{input_text}" if input_text else instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_path", required=True,
                         help="Output of generate_candidates.py run against SFT_MODEL")
    parser.add_argument("--output", default="data/mt_data/dpo_data/stage1_pairs.json")
    parser.add_argument("--min_prediction_chars", type=int, default=1,
                         help="Drop pairs whose model output is empty or degenerate")
    parser.add_argument("--max_chrf", type=float, default=75.0,
                         help="Skip pairs whose prediction already scores at or above this "
                              "chrF++ against the reference; 100 disables the filter")
    args = parser.parse_args()

    # Step 1: read the model outputs generated earlier by generate_candidates.py.
    with open(args.candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    # Step 2: build one preference pair per candidate. The human reference is always the
    # "chosen" side and the model's own output is the "rejected" side.
    pairs = []
    n_dropped = 0
    for c in candidates:
        chosen = c["reference"].strip()
        rejected = c["prediction"].strip()
        # An empty or identical prediction gives no contrast to learn from.
        if len(rejected) < args.min_prediction_chars or rejected == chosen:
            n_dropped += 1
            continue
        # Measure how close the two sides are, and drop the pair if they nearly agree.
        score = sacrebleu.sentence_chrf(rejected, [chosen], word_order=2).score
        if score >= args.max_chrf:
            n_dropped += 1
            continue
        pairs.append({
            "prompt": build_prompt(c["instruction"], c["input"]),
            "chosen": chosen,
            "rejected": rejected,
            "rejected_chrf": score,
        })

    # Step 3: write the pairs out, ready for DPO training.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(pairs)} Stage 1 pairs to {args.output} ({n_dropped} dropped as too-close/degenerate)")
