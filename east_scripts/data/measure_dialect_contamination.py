"""
Quick MSA-contamination check for a predictions.json: counts how many predictions contain an
MSA function word listed in dialect_lexicon.

The table is small and will miss MSA forms it does not know, so read the number as a
directional signal (did contamination drop between stages?), not an absolute score.
"""
import argparse
import json

from dialect_lexicon import DIALECT_TO_MSA_TABLES

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True,
                         help="A predictions.json or a generate_candidates.py output")
    parser.add_argument("--dialect", default="EG")
    args = parser.parse_args()

    # Step 1: collect the MSA words to look for, which are the values of the dialect table.
    table = DIALECT_TO_MSA_TABLES.get(args.dialect)
    if table is None:
        raise ValueError(f"No MSA table for dialect {args.dialect!r}")
    msa_words = set(table.values())

    with open(args.predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Step 2: count a prediction as contaminated if it shares any whole word with that set.
    n_contaminated = 0
    for row in data:
        pred = row.get("prediction", "")
        pred_words = set(pred.split())
        if pred_words & msa_words:
            n_contaminated += 1

    rate = n_contaminated / len(data) * 100 if data else 0.0
    print(f"MSA-marker contamination: {n_contaminated}/{len(data)} ({rate:.1f}%) predictions contain an MSA form")
