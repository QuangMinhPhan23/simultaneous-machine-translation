"""
DPO Stage 2 pairs: the human reference in the right dialect (chosen) vs. the same text
rendered with another dialect's function-word markers (rejected), via
dialect_lexicon.swap_dialect().

Swapping markers on the same text keeps the pair minimal, so only the dialect differs and
not the content. Pairs where nothing was swapped are skipped, since both sides would match.
"""
import argparse
import json
import os

from dialect_lexicon import swap_dialect

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/mt_data/train_data/Arabic-EG-SiMT-OMT.json",
                         help="Alpaca-schema OMT examples for the source dialect")
    parser.add_argument("--src_dialect", default="EG")
    parser.add_argument("--dst_dialect", default="LB")
    parser.add_argument("--output", default="data/mt_data/dpo_data/stage2_pairs.json")
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    examples = [ex for ex in examples if "latency" not in ex]  # OMT rows only

    pairs = []
    for ex in examples:
        chosen = ex["output"].strip()
        rejected, n_hits = swap_dialect(chosen, args.src_dialect, args.dst_dialect)
        if n_hits == 0:
            continue
        pairs.append({
            "prompt": f"{ex['instruction']}\n{ex['input']}" if ex.get("input") else ex["instruction"],
            "chosen": chosen,
            "rejected": rejected,
        })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(pairs)} Stage 2 pairs ({args.src_dialect} vs {args.dst_dialect} markers) to {args.output}")
