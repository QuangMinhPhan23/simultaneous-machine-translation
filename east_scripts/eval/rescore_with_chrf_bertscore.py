"""
Re-scores existing prediction.json files with chrF++ and optionally BERTScore,
without re-running the model. BLEU is harsh on dialectal Arabic spelling
variation, so these two give a fairer picture.

Input: one cell (--cell) or every cell (--all) under a results root.
Output: a printed table; pass --write to also update the JSON files in place.
"""
import argparse
import glob
import json
import os

import sacrebleu


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chrf_pp(hyps, refs):
    """chrF++ = chrF with word_order=2 (sacrebleu convention).

    It matches character n-grams plus word bigrams, so a differently spelled but correct word
    still earns partial credit. Returns the corpus score and the per-sentence scores."""
    corpus = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    per_sentence = [
        sacrebleu.sentence_chrf(h, [r], word_order=2).score for h, r in zip(hyps, refs)
    ]
    return corpus, per_sentence


def bert_score_arabic(hyps, refs, model_type):
    """BERTScore F1, scaled x100. It embeds both sides with a multilingual model and matches
    each output word to its closest reference word, so paraphrases are not punished as hard as
    BLEU punishes them. The corpus number here is the mean of the per-sentence F1s."""
    from bert_score import score as bertscore_fn  # imported here so it stays optional

    P, R, F1 = bertscore_fn(hyps, refs, model_type=model_type, verbose=False)
    per_sentence = [f.item() * 100 for f in F1]
    corpus = sum(per_sentence) / len(per_sentence)
    return corpus, per_sentence


def rescore_cell(pred_path, do_bertscore, bertscore_model, write):
    """Score one cell's saved predictions and return its summary row.

    A "cell" is one model / variant / latency folder. With --write, the new scores are also
    stored back into that folder's prediction.json and results.json."""
    preds = load_json(pred_path)
    hyps = [p["prediction"] for p in preds]
    refs = [p["reference"] for p in preds]

    chrf_corpus, chrf_sent = chrf_pp(hyps, refs)
    for p, s in zip(preds, chrf_sent):
        p["chrF++"] = s

    row = {"n": len(preds), "chrF++": chrf_corpus}

    if do_bertscore:
        bs_corpus, bs_sent = bert_score_arabic(hyps, refs, bertscore_model)
        for p, s in zip(preds, bs_sent):
            p["BERTScore_F1"] = s
        row["BERTScore_F1"] = bs_corpus

    # Write the per-sentence scores back into prediction.json, and merge the corpus scores into
    # results.json without dropping the metrics that are already in there.
    if write:
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=4)
        results_path = os.path.join(os.path.dirname(pred_path), "results.json")
        try:
            results = load_json(results_path)
        except FileNotFoundError:
            results = {}
        results["chrF++"] = chrf_corpus
        if do_bertscore:
            results["BERTScore_F1"] = row["BERTScore_F1"]
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    return row


def main():
    """Pick the cells to re-score, score each one, and print them as a table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/granularity_comparison_alldomains")
    parser.add_argument("--cell", default=None,
                         help="Single cell relative to results_root, e.g. nilechat/chunk-llama/high")
    parser.add_argument("--all", action="store_true",
                         help="Re-score every */*/*/prediction.json under results_root")
    parser.add_argument("--bertscore", action="store_true",
                         help="Also compute BERTScore (needs the bert_score package)")
    parser.add_argument("--bertscore_model", default="bert-base-multilingual-cased",
                         help="BERTScore embedding model (mBERT works for Arabic)")
    parser.add_argument("--write", action="store_true",
                         help="Update each cell's prediction.json + results.json in place")
    args = parser.parse_args()

    # One named cell, or every model/variant/latency folder under the results root.
    if args.cell:
        pred_paths = [os.path.join(args.results_root, args.cell, "prediction.json")]
    elif args.all:
        pred_paths = sorted(glob.glob(os.path.join(args.results_root, "*", "*", "*", "prediction.json")))
    else:
        parser.error("pass either --cell <model/variant/latency> or --all")

    if not pred_paths:
        print(f"No prediction.json found under {args.results_root}")
        return

    header = ["cell", "n", "chrF++"] + (["BERTScore_F1"] if args.bertscore else [])
    print("\t".join(header))
    for pred_path in pred_paths:
        if not os.path.exists(pred_path):
            print(f"MISSING\t{pred_path}")
            continue
        cell = os.path.relpath(os.path.dirname(pred_path), args.results_root)
        row = rescore_cell(pred_path, args.bertscore, args.bertscore_model, args.write)
        vals = [cell, str(row["n"]), f'{row["chrF++"]:.2f}']
        if args.bertscore:
            vals.append(f'{row["BERTScore_F1"]:.2f}')
        print("\t".join(vals))

    if args.write:
        print("\n(updated prediction.json + results.json in place for each cell)")


if __name__ == "__main__":
    main()
