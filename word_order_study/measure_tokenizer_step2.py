"""Step 2: tokenizer analysis for the word-order study.

Measures how efficiently the Llama-3-8B tokenizer (the EAST base's tokenizer) encodes each language's
target side. A language that needs ~2x the tokens for the same content gets an artificially inflated
token-based Average Lagging; this quantifies that confound. Reports chars/token, tokens/sentence, and
tokens/word for the target (and English source as reference) over the first --n train sentences.

Tokenizer: NousResearch/Meta-Llama-3-8B-Instruct, an ungated identical mirror of the official Llama-3
(same 128k vocab); falls back to the gated official repo if an HF token is present.

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
    """Return the first candidate tokenizer that loads.

    The official repo is gated, so it only works with an HF token; otherwise the identical
    ungated mirror is used. Both have the same 128k vocabulary."""
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
    """Tokenize a list of sentences and return the fertility numbers for them.

    Totals are summed over all the sentences first, then divided, so long sentences count more
    than short ones. Words are whitespace-split."""
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
    """Measure how many tokens each language needs, and write the comparison table.

    The last table is the important one: it shows how much a token-based latency score would be
    inflated for the languages the tokenizer splits more finely."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    # Step 1: load the tokenizer once and reuse it for every language.
    tok = load_tokenizer()

    # Step 2: measure both sides of each language's split. The English side is the same kind of
    # text everywhere, so it acts as a reference the target numbers can be read against.
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

    # Step 4: AL-inflation index - tokens/sentence relative to the least-fragmented language.
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

    # Step 5: print the report and save the same text as a markdown file.
    out = "# Step 2 - Tokenizer Fertility\n\n" + fmt_target() + "\n\n" + fmt_source() + "\n\n" + "\n".join(idx) + "\n"
    print(out)
    with open(os.path.join(DATA, "tokenizer_fertility.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print(f"Wrote {DATA}/tokenizer_fertility.md")


if __name__ == "__main__":
    main()
