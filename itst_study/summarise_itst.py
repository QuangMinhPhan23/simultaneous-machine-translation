"""Print results tables from results/<dataset>/results.json.

Tables: threshold sweep, quality at matched AL, degradation (high to low latency).

Usage:
  python itst_study/summarise_itst.py
  python itst_study/summarise_itst.py --metric bleu_tokenized
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

ORDER = {"vietnamese": "SVO", "egyptian": "SVO (dialect)", "saudi": "VSO-leaning",
         "saudi-matched": "VSO-leaning", "msa": "VSO", "korean": "SOV", "paper-envi": "SVO"}
# Latencies to compare at, in source tokens of Average Lagging.
AL_POINTS = [2.0, 4.0, 6.0, 8.0]


def load_all(datasets=None):
    """Read every results.json under results/, newest layout only."""
    out = {}
    if not os.path.isdir(RESULTS):
        return out
    for name in sorted(os.listdir(RESULTS)):
        path = os.path.join(RESULTS, name, "results.json")
        if not os.path.exists(path):
            continue
        if datasets and name not in datasets:
            continue
        with open(path, encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def interp_at_al(rows, target_al, metric):
    """Score at a given AL, linearly interpolated between sweep points."""
    pts = sorted([(r["AL"], r[metric]) for r in rows
                  if r.get("AL") is not None and r.get(metric) is not None])
    if len(pts) < 2 or target_al < pts[0][0] or target_al > pts[-1][0]:
        return None
    for (a1, q1), (a2, q2) in zip(pts, pts[1:]):
        if a1 <= target_al <= a2:
            if a2 == a1:
                return q1
            return q1 + (q2 - q1) * (target_al - a1) / (a2 - a1)
    return None


def table_sweep(data, metric):
    """One block per dataset: every threshold with its latency and quality."""
    lines = []
    for name, res in data.items():
        lines.append(f"\n**{name}** ({ORDER.get(name, '?')}), scored by `{res['scorer']}`\n")
        lines.append("| delta | CW | AP | AL | DAL | " + metric + " |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in res["rows"]:
            def g(k):
                v = r.get(k)
                return "pending" if v is None else v
            lines.append(f"| {r['delta']} | {g('CW')} | {g('AP')} | {g('AL')} | {g('DAL')} | "
                         f"{g(metric)} |")
    return lines


def table_matched(data, metric):
    """Quality at the same Average Lagging across languages."""
    lines = ["", "| Language | Word order | " +
             " | ".join(f"AL={a}" for a in AL_POINTS) + " |",
             "|---|---|" + "---:|" * len(AL_POINTS)]
    for name, res in data.items():
        cells = []
        for a in AL_POINTS:
            v = interp_at_al(res["rows"], a, metric)
            cells.append("pending" if v is None else f"{v:.2f}")
        lines.append(f"| {name} | {ORDER.get(name, '?')} | " + " | ".join(cells) + " |")
    return lines


def table_degradation(data, metric):
    """Drop from the slowest sweep point to the fastest, in points and as a percentage."""
    lines = ["", f"| Language | Word order | {metric} at lowest AL | at highest AL | drop | "
                 "degradation |",
             "|---|---|---:|---:|---:|---:|"]
    for name, res in data.items():
        pts = sorted([(r["AL"], r[metric]) for r in res["rows"]
                      if r.get("AL") is not None and r.get(metric) is not None])
        if len(pts) < 2:
            lines.append(f"| {name} | {ORDER.get(name, '?')} | pending | pending | pending | "
                         "pending |")
            continue
        (lo_al, lo_q), (hi_al, hi_q) = pts[0], pts[-1]
        drop = hi_q - lo_q
        pct = 100.0 * drop / hi_q if hi_q else 0.0
        lines.append(f"| {name} | {ORDER.get(name, '?')} | {lo_q:.2f} (AL={lo_al}) | "
                     f"{hi_q:.2f} (AL={hi_al}) | {drop:.2f} | {pct:.1f}% |")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="chrf2",
                    help="chrf2, bleu, or bleu_tokenized for the paper track")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--al_points", nargs="+", type=float, default=AL_POINTS,
                    help="Average Lagging values to compare quality at")
    args = ap.parse_args()
    AL_POINTS = args.al_points

    data = load_all(args.datasets)
    if not data:
        raise SystemExit(f"no results.json under {RESULTS}")

    out = ["## Threshold sweep"] + table_sweep(data, args.metric)
    out += ["", "## Quality at matched latency"] + table_matched(data, args.metric)
    out += ["", "## Degradation from high to low latency"] + table_degradation(data, args.metric)
    print("\n".join(out))
