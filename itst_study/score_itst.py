"""Add quality scores (BLEU, chrF++) to the results.json from eval_itst.py.

Runs in ~/venvs/itst-score (sacrebleu 2.x) because the fairseq env needs 1.x.

Usage:
  ~/venvs/itst-score/bin/python itst_study/score_itst.py --dataset paper-envi --metric multibleu
  ~/venvs/itst-score/bin/python itst_study/score_itst.py --dataset korean
"""
import argparse
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ITST_REPO = os.environ.get("ITST_REPO", os.path.join(HERE, "ITST"))
RESULTS = os.path.join(HERE, "results")


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def score_multibleu(hyp_path, ref_path):
    """Tokenized BLEU, case-insensitive, with the paper's multi-bleu.perl."""
    script = os.path.join(ITST_REPO, "multi-bleu.perl")
    with open(hyp_path, encoding="utf-8") as f:
        p = subprocess.run(["perl", script, "-lc", ref_path],
                           stdin=f, capture_output=True, text=True)
    m = re.search(r"BLEU\s*=\s*([\d.]+)", p.stdout)
    if not m:
        print("  multi-bleu.perl said:", p.stdout.strip(), p.stderr.strip())
        return {"bleu_tokenized": None}
    return {"bleu_tokenized": float(m.group(1))}


def score_chrf(hyp_path, ref_path):
    """Detokenized BLEU and chrF++."""
    import sacrebleu
    hyps, refs = read_lines(hyp_path), read_lines(ref_path)
    if len(hyps) != len(refs):
        print(f"  WARNING: {len(hyps)} hypotheses vs {len(refs)} references, "
              "scoring the common prefix")
    n = min(len(hyps), len(refs))
    hyps, refs = hyps[:n], refs[:n]
    return {"bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrf2": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
            "n_scored": n}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metric", choices=["chrf", "multibleu"], default="chrf")
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS, args.dataset)
    path = os.path.join(out_dir, "results.json")
    with open(path, encoding="utf-8") as f:
        res = json.load(f)

    ref = res["reference"]
    scorer = score_multibleu if args.metric == "multibleu" else score_chrf
    for row in res["rows"]:
        hyp = os.path.join(out_dir, row["hyp_file"])
        row.update(scorer(hyp, ref))
        print(f"  delta={row['delta']:<5} AL={row.get('AL')}  "
              + "  ".join(f"{k}={v}" for k, v in row.items()
                          if k in ("bleu", "chrf2", "bleu_tokenized")))

    res["scorer"] = args.metric
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated {path}")
