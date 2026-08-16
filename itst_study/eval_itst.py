"""Decode at each delta threshold and record latency numbers.

Decoding only -- score_itst.py adds quality scores in a separate venv
(fairseq needs sacrebleu 1.x, chrF++ needs 2.x).

Usage:
  python itst_study/eval_itst.py --dataset paper-envi --thresholds 0.1 0.2 0.3 0.4 0.5
  python itst_study/eval_itst.py --dataset saudi --spm_from msa
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ITST_REPO = os.environ.get("ITST_REPO", os.path.join(HERE, "ITST"))
BIN = os.path.join(HERE, "data-bin")
# Checkpoints sit on scratch, set by env.sh, because they do not fit in the home quota.
CKPT = os.environ.get("ITST_CKPT", os.path.join(HERE, "checkpoints"))
RESULTS = os.path.join(HERE, "results")

EXT = {"vietnamese": "vi", "msa": "ar", "korean": "ko", "saudi": "ar", "egyptian": "ar",
       "saudi-matched": "ar", "paper-envi": "vi"}


def run(cmd, out_path=None):
    """Run a command, sending stdout to a file when one is given."""
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)


def average_checkpoints(save_dir, n=5):
    """Average the last n saved checkpoints, which is what the paper decodes with."""
    out = os.path.join(save_dir, "average-model.pt")
    if os.path.exists(out):
        print(f"  reusing {out}")
        return out
    last = os.path.join(save_dir, "checkpoint_last.pt")
    run([sys.executable, os.path.join(ITST_REPO, "scripts", "average_checkpoints.py"),
         "--inputs", save_dir, "--num-update-checkpoints", str(n),
         "--output", out, "--last_file", last])
    return out


def parse_generate(path):
    """Extract latency averages and hypotheses from a sim_generate log."""
    lat, hyps = {}, {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^(CW|AP|AL|DAL) score:\s+(-?[\d.]+)", line)
            if m:
                lat[m.group(1)] = round(float(m.group(2)), 2)
                continue
            if line.startswith("H-"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    hyps[int(parts[0][2:])] = parts[2]
    return lat, [hyps[k] for k in sorted(hyps)]


def detok_spm(lang, lines, out_path):
    """Undo SentencePiece so the output can be scored against the plain reference."""
    import sentencepiece as spm
    model = os.path.join(HERE, "spm", lang, "spm.model")
    sp = spm.SentencePieceProcessor(model_file=model)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(sp.decode(line.split()) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="names the checkpoint and the results folder")
    ap.add_argument("--data_from", choices=list(EXT), default=None,
                    help="dataset whose data-bin and test reference to use, when the checkpoint "
                         "is a variant run trained on another dataset's data")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--spm_from", default=None,
                    help="language whose SentencePiece model decoded this data (dialects use msa)")
    args = ap.parse_args()

    ds = args.dataset
    src = args.data_from or ds
    if src not in EXT:
        sys.exit(f"unknown data source {src}; pass --data_from")
    ext = EXT[src]
    data_bin = os.path.join(BIN, src)
    save_dir = os.path.join(CKPT, ds)
    out_dir = os.path.join(RESULTS, ds)
    os.makedirs(out_dir, exist_ok=True)

    model = average_checkpoints(save_dir)

    # The reference is the plain text test set, not the encoded one.
    ref = os.path.join(HERE, "data", src, f"test.{ext}")

    rows = []
    for delta in args.thresholds:
        tag = f"delta{delta}"
        log = os.path.join(out_dir, f"pred.{tag}.out")
        print(f"\n=== {ds} delta={delta} ===", flush=True)
        cmd = [sys.executable, os.path.join(ITST_REPO, "fairseq_cli", "sim_generate.py"),
               data_bin, "--path", model, "--batch-size", "1", "--beam", "1",
               "--left-pad-source", "--fp16",
               "--itst-decoding", "--itst-test-threshold", str(delta)]
        run(cmd, out_path=log)

        lat, hyps = parse_generate(log)
        # sim_generate can exit 0 with no output, so check explicitly.
        if not hyps:
            sys.exit(f"{log} has no H- lines: sim_generate produced nothing for delta={delta}")
        if not lat:
            sys.exit(f"{log} has no latency lines for delta={delta}")

        hyp_path = os.path.join(out_dir, f"pred.{tag}.txt")
        if src == "paper-envi":
            with open(hyp_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("".join(h + "\n" for h in hyps))
        else:
            detok_spm(args.spm_from or src, hyps, hyp_path)

        row = {"delta": delta, **lat, "n_hyp": len(hyps),
               "hyp_file": os.path.basename(hyp_path)}
        print("  ", row, flush=True)
        rows.append(row)

        # Write after each threshold so partial results survive a walltime kill.
        res = {"dataset": ds, "model": model, "reference": ref, "rows": rows}
        path = os.path.join(out_dir, "results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {path} - now run score_itst.py to add the quality scores")


if __name__ == "__main__":
    main()
