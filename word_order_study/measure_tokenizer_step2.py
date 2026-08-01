"""Step 2: tokenizer analysis for the word-order study.

Measures how many Llama-3 tokens each language needs for the same content, over the first --n
sentences of a split. Reports chars/token, tokens/sentence and tokens/word for the target side,
the English side as a reference, and how much a token-based latency score is inflated per language.
Writes data/tokenizer_fertility.md.

Usage:
  python word_order_study/measure_tokenizer_step2.py [--n 1000 --split train]
"""
import argparse
import os
import sys

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

LANGS = [  # (name, tgt_ext, script)
    ("vietnamese", "vi", "Latin + diacritics"),
    ("egyptian",   "ar", "Arabic"),
    ("saudi",      "ar", "Arabic"),
    ("msa",        "ar", "Arabic"),
    ("korean",     "ko", "Hangul"),
]

TOKENIZER_CANDIDATES = [
    "meta-llama/Meta-Llama-3-8B-Instruct",   # official (gated) - used if HF token present
    "NousResearch/Meta-Llama-3-8B-Instruct",  # ungated identical mirror
]


def load_tokenizer():
    """Return the first candidate tokenizer that loads."""
    last = None
    for m in TOKENIZER_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(m)
            print(f"Tokenizer: {m} (vocab {tok.vocab_size})\n")
            return tok
        except Exception as e:
            last = f"{m}: {type(e).__name__}"
    raise SystemExit(f"Could not load any Llama-3 tokenizer ({last})")


def read_lines(path, n):
    """Read at most n non-blank lines from a file."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    return lines[:n]


def measure(tok, texts):
    """Tokenize the sentences and return the fertility ratios, summed first then divided."""
    chars = sum(len(t) for t in texts)
    words = sum(len(t.split()) for t in texts)
    toks = sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
    n = len(texts)
    return {
        "n": n,
        "chars_per_token": chars / toks if toks else 0.0,
        "tokens_per_sent": toks / n if n else 0.0,
        "tokens_per_word": toks / words if words else 0.0,
        "chars_per_sent": chars / n if n else 0.0,
    }


def main():
    """Measure how many tokens each language needs, and write the comparison tables."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    # Step 1: load the tokenizer once and reuse it for every language.
    tok = load_tokenizer()

    # Step 2: measure both sides of each language. English is the same everywhere, so it is a reference.
    tgt_rows, src_rows = [], []
    for name, ext, script in LANGS:
        tgt = read_lines(os.path.join(DATA, name, f"{args.split}.{ext}"), args.n)
        src = read_lines(os.path.join(DATA, name, f"{args.split}.en"), args.n)
        tgt_rows.append((name, script, measure(tok, tgt)))
        src_rows.append((name, measure(tok, src)))

    # Step 3: turn the two sets of rows into markdown tables.
    def fmt_target():
        L = ["## Target side (Llama-3-8B-Instruct tokenizer)", "",
             "| Language | Script | n | chars/token | tokens/sentence | tokens/word | chars/sentence |",
             "|---|---|---:|---:|---:|---:|---:|"]
        for name, script, s in tgt_rows:
            L.append(f"| {name} | {script} | {s['n']} | {s['chars_per_token']:.3f} | "
                     f"{s['tokens_per_sent']:.1f} | {s['tokens_per_word']:.3f} | {s['chars_per_sent']:.1f} |")
        return "\n".join(L)

    def fmt_source():
        L = ["## English source side (read-side reference, same rows)", "",
             "| Language | chars/token | tokens/sentence | tokens/word |",
             "|---|---:|---:|---:|"]
        for name, s in src_rows:
            L.append(f"| {name} | {s['chars_per_token']:.3f} | {s['tokens_per_sent']:.1f} | "
                     f"{s['tokens_per_word']:.3f} |")
        return "\n".join(L)

    # Step 4: AL-inflation index, each language's tokens/sentence over the lowest one.
    min_tps = min(s["tokens_per_sent"] for _, _, s in tgt_rows)
    idx = ["## Token-based AL inflation index (target tokens/sentence ÷ lowest)",
           "",
           "Higher = the same 15-30-word English content is emitted as more target tokens, so a "
           "token-defined AL is inflated for that language. Report AL in **words/chars** too.",
           "",
           "| Language | tokens/sentence | AL-inflation vs lowest |",
           "|---|---:|---:|"]
    for name, script, s in tgt_rows:
        idx.append(f"| {name} | {s['tokens_per_sent']:.1f} | {s['tokens_per_sent']/min_tps:.2f}x |")

    # Step 5: print the report and save the same text as markdown.
    out = "# Step 2 - Tokenizer Fertility\n\n" + fmt_target() + "\n\n" + fmt_source() + "\n\n" + "\n".join(idx) + "\n"
    print(out)
    with open(os.path.join(DATA, "tokenizer_fertility.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print(f"Wrote {DATA}/tokenizer_fertility.md")


if __name__ == "__main__":
    main()
