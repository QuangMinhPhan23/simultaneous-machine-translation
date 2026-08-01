"""Re-score finished predictions with COMET-KIWI, a metric that does not read the reference.

Reads the prediction.json files already written by simuleval_standalone.py, so nothing is decoded
again. Scores each (system, latency) cell from the source and the output only, prints a table with
the high-to-low latency drop, and with --write stores the score in each results.json.

Usage:
  python word_order_study/rescore_kiwi.py --result_root results/wordorder
"""
import argparse
import glob
import json
import os
import sys


def load_cells(result_root):
    """Find every finished (system, latency) cell that has predictions."""
    cells = []
    for pred_path in sorted(glob.glob(os.path.join(result_root, "*", "*", "prediction.json"))):
        latency = os.path.basename(os.path.dirname(pred_path))
        system = os.path.basename(os.path.dirname(os.path.dirname(pred_path)))
        if "smoke" in system:
            continue
        cells.append((system, latency, pred_path))
    return cells


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_root", default="results/wordorder")
    ap.add_argument("--model", default="Unbabel/wmt22-cometkiwi-da",
                    help="reference-free COMET model; needs a HF token, the repo is gated")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--write", action="store_true", help="also store the score in results.json")
    args = ap.parse_args()

    cells = load_cells(args.result_root)
    if not cells:
        raise SystemExit(f"no prediction.json under {args.result_root}")
    print(f"Found {len(cells)} cells")

    from comet import download_model, load_from_checkpoint
    import torch
    print(f"Loading {args.model}")
    model = load_from_checkpoint(download_model(args.model))
    gpus = 1 if torch.cuda.is_available() else 0

    rows = []
    for system, latency, pred_path in cells:
        with open(pred_path, encoding="utf-8") as f:
            preds = json.load(f)
        # Source and output only, the reference is not passed.
        data = [{"src": p["source"], "mt": p["prediction"]} for p in preds]
        out = model.predict(data, batch_size=args.batch_size, gpus=gpus)
        kiwi = out.system_score * 100
        rows.append((system, latency, kiwi, len(data)))
        print(f"  {system:24} {latency:8} kiwi={kiwi:6.2f}  (n={len(data)})")

        if args.write:
            res_path = os.path.join(os.path.dirname(pred_path), "results.json")
            res = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else {}
            res["COMET-KIWI"] = kiwi
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

    # How much each system loses going from high to low latency.
    print("\nDegradation on COMET-KIWI  (kiwi_high - kiwi_low) / kiwi_high")
    by_system = {}
    for system, latency, kiwi, _ in rows:
        by_system.setdefault(system, {})[latency] = kiwi
    for system in sorted(by_system):
        d = by_system[system]
        if "low" in d and "high" in d:
            drop = (d["high"] - d["low"]) / d["high"] * 100
            print(f"  {system:24} low={d['low']:6.2f} high={d['high']:6.2f}  drop={drop:5.1f}%")


if __name__ == "__main__":
    main()
