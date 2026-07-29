"""
Collects the per-latency results.json files from an eval run into one table.

Input: one or more result roots, each optionally written as label=path.
Output: a printed table of BLEU / spBLEU / chrF++ / COMET / BERTScore / AL / LAAL.
Missing latency folders are skipped silently.
"""
import argparse
import json
import os

LATENCIES = ["low", "low-medium", "medium", "medium-high", "high"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_root", nargs="+", default=["results/alexandria_eg_commerce"],
        help="One or more result roots, optionally as label=path",
    )
    args = parser.parse_args()

    roots = []
    for entry in args.result_root:
        if "=" in entry:
            label, path = entry.split("=", 1)
        else:
            label, path = os.path.basename(os.path.normpath(entry)), entry
        roots.append((label, path))

    rows = []
    for label, root in roots:
        for latency in LATENCIES:
            result_file = os.path.join(root, latency, "results.json")
            if not os.path.exists(result_file):
                continue
            with open(result_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            results["config"] = label
            results["latency"] = latency
            rows.append(results)

    show_config_col = len(roots) > 1
    # Any metric missing from a results.json just prints as an empty cell.
    header = (["config"] if show_config_col else []) + [
        "latency", "BLEU", "spBLEU", "chrF++", "COMET", "BERTScore", "BLEURT", "AL", "LAAL"]
    widths = ([18] if show_config_col else []) + [14, 8, 8, 8, 8, 10, 8, 8, 8]

    def fmt_cell(value):
        return f"{value:.2f}" if isinstance(value, float) else str(value) if value is not None else ""

    def fmt_row(values):
        return "".join(f"{fmt_cell(v):<{w}}" for v, w in zip(values, widths))

    print(fmt_row(header))
    for row in rows:
        print(fmt_row(row.get(col) for col in header))
