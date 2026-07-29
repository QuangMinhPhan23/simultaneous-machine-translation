"""
Rescores an existing prediction.json against an MSA reference as well as the
original dialect reference, to show how MSA-leaning the output is.

Input: a prediction.json and a matching MSA reference file.
Output: a printed BLEU / chrF++ / COMET table, dialect vs MSA.
No model inference is needed, only the saved predictions.
"""
import argparse
import json

import sacrebleu
import torch
from comet import download_model, load_from_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", required=True,
                         help="An existing prediction.json from one of the eval scripts")
    parser.add_argument("--msa_reference_path", required=True,
                         help="MSA reference file, matched to predictions by 'index'")
    parser.add_argument("--comet_ckpt_path", default=None,
                         help="COMET checkpoint path; omit to auto-download wmt22-comet-da")
    args = parser.parse_args()

    # Step 1: load the saved predictions and the MSA references, keyed by index.
    with open(args.prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    with open(args.msa_reference_path, "r", encoding="utf-8") as f:
        msa_refs = {r["index"]: r["msa_reference"] for r in json.load(f)}

    # Step 2: keep only the predictions that have an MSA reference, and build four parallel
    # lists so the same output can be scored against both reference sets.
    hypos, dialect_refs, msa_ref_list, sources = [], [], [], []
    for p in predictions:
        idx = p["index"]
        if idx not in msa_refs:
            continue
        hypos.append(p["prediction"])
        dialect_refs.append(p["reference"])
        msa_ref_list.append(msa_refs[idx])
        sources.append(p["source"])

    print(f"Scoring {len(hypos)} matched predictions")

    # Step 3: surface metrics against each reference set. BLEU counts matching word n-grams,
    # chrF++ matches character n-grams plus word bigrams, which is fairer to spelling variation.
    # A higher score against MSA than against the dialect means the output leans formal.
    bleu_dialect = sacrebleu.corpus_bleu(hypos, [dialect_refs]).score
    bleu_msa = sacrebleu.corpus_bleu(hypos, [msa_ref_list]).score
    chrf_dialect = sacrebleu.corpus_chrf(hypos, [dialect_refs], word_order=2).score
    chrf_msa = sacrebleu.corpus_chrf(hypos, [msa_ref_list], word_order=2).score

    # Step 4: the same comparison with COMET, a trained neural scorer that reads source, output
    # and reference together. It is run twice, once per reference set.
    comet_ckpt = args.comet_ckpt_path or download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_ckpt)
    gpus = 1 if torch.cuda.is_available() else 0
    data_dialect = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypos, dialect_refs)]
    data_msa = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypos, msa_ref_list)]
    comet_dialect = comet_model.predict(data_dialect, batch_size=256, gpus=gpus).system_score * 100
    comet_msa = comet_model.predict(data_msa, batch_size=256, gpus=gpus).system_score * 100

    # Step 5: print the two columns side by side.
    print(f"{'Metric':<10}{'vs Dialect':<14}{'vs MSA':<14}")
    print(f"{'BLEU':<10}{bleu_dialect:<14.2f}{bleu_msa:<14.2f}")
    print(f"{'chrF++':<10}{chrf_dialect:<14.2f}{chrf_msa:<14.2f}")
    print(f"{'COMET':<10}{comet_dialect:<14.2f}{comet_msa:<14.2f}")
