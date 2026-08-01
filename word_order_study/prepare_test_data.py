"""Step 5 prep: turn the held-out test splits into the JSON the evaluator reads.

Reads data/<language>/test.en and test.<ext> and writes data/<language>/test.eval.json with the
{"source", "reference", "src_lang", "tgt_lang"} fields simuleval_standalone.py expects.

Usage:
  python word_order_study/prepare_test_data.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)

import chunk_prompts

EXT = {"vietnamese": "vi", "msa": "ar", "korean": "ko", "saudi": "ar", "egyptian": "ar"}


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def build(language, split):
    """Pair up the source and reference lines for one language."""
    ext = EXT[language]
    src = read_lines(os.path.join(DATA, language, f"{split}.en"))
    tgt = read_lines(os.path.join(DATA, language, f"{split}.{ext}"))
    if len(src) != len(tgt):
        raise SystemExit(f"{language}: line counts differ, {len(src)} vs {len(tgt)}")
    tgt_lang = chunk_prompts.TGT_LANG_NAME[language]
    rows = []
    for s, t in zip(src, tgt):
        s, t = s.strip(), t.strip()
        if s and t:
            rows.append({"source": s, "reference": t, "src_lang": "English", "tgt_lang": tgt_lang})
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", default=list(EXT))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    for language in args.languages:
        rows = build(language, args.split)
        out = os.path.join(DATA, language, f"{args.split}.eval.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"  {language}: {len(rows)} sentences -> {out}")


if __name__ == "__main__":
    main()
