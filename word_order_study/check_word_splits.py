"""Check chunk quality: words cut in half, and steps that write only punctuation.

Vietnamese whitespace separates syllables, not words, so splitting on whitespace can end a chunk
inside a word: "dan chu" (democracy) becomes "nen dan" | "chu". For each chunks file this joins the
target chunks back together, segments them with segment.atomic_units, and checks that every chunk
boundary lands on a word end. Prints one row per file and latency.

Usage:
  python word_order_study/check_word_splits.py --files chunks-generic.json chunks-aligned.json
"""
import argparse
import json
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)

from segment import atomic_units

LATENCIES = ["low", "medium", "high"]


def word_end_positions(text, language):
    """Syllable indices where a word ends. A chunk boundary is only legal at one of these."""
    positions, n = set(), 0
    for unit in atomic_units(text, language):
        n += len(unit.split())
        positions.add(n)
    return positions


def count_cuts(entries, latency, language):
    """Return (sentences with a cut word, total), then the same two counts for fallback sentences."""
    bad = total = bad_fb = total_fb = 0
    for e in entries:
        chunks = e.get("chunks", {}).get(latency)
        if not chunks:
            continue
        target_chunks = chunks["target_chunks"]
        legal = word_end_positions(" ".join(target_chunks), language)
        n, cut = 0, False
        for chunk in target_chunks[:-1]:          # the last boundary is the sentence end
            n += len(chunk.split())
            if n not in legal:
                cut = True
                break
        total += 1
        bad += cut
        if e.get("fallback_by_latency", {}).get(latency):
            total_fb += 1
            bad_fb += cut
    return bad, total, bad_fb, total_fb


def count_punct_only(entries, latency):
    """Return (steps whose target is only punctuation, total steps) for one latency."""
    punct = total = 0
    for e in entries:
        chunks = e.get("chunks", {}).get(latency)
        if not chunks:
            continue
        for target in chunks["target_chunks"]:
            total += 1
            stripped = target.strip()
            if stripped and all(unicodedata.category(c)[0] in ("P", "S") or c.isspace()
                                for c in stripped):
                punct += 1
    return punct, total


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="vietnamese")
    ap.add_argument("--files", nargs="+", default=["chunks-generic.json", "chunks-specific.json",
                                                   "chunks-aligned.json"])
    args = ap.parse_args()

    print(f"{args.language}: sentences where a chunk boundary cuts a word in half\n")
    print(f"  {'file':28} {'latency':8} {'words cut':>18} {'fallback only':>18}"
          f"  {'punctuation-only steps':>24}")
    for fname in args.files:
        path = os.path.join(DATA, args.language, fname)
        if not os.path.exists(path):
            print(f"  {fname:28} (missing)")
            continue
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        for latency in LATENCIES:
            bad, total, bad_fb, total_fb = count_cuts(entries, latency, args.language)
            pct = 100.0 * bad / max(1, total)
            fb = f"{bad_fb}/{total_fb} ({100.0 * bad_fb / max(1, total_fb):.1f}%)" if total_fb else "-"
            punct, steps = count_punct_only(entries, latency)
            ppct = 100.0 * punct / max(1, steps)
            print(f"  {fname:28} {latency:8} {bad:>6}/{total} ({pct:>5.1f}%) {fb:>18}"
                  f"  {punct:>6}/{steps} ({ppct:>4.1f}%)")


if __name__ == "__main__":
    main()
