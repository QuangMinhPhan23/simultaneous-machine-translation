"""Build parallel corpora for ITST training from Hugging Face sources.

Dev/test come from word_order_study unchanged. --max_train caps all languages to the same size
so corpus size cannot look like a word-order effect.

Usage:
  python itst_study/source_itst_data.py --languages vietnamese msa korean --max_train 130000
  python itst_study/source_itst_data.py --languages saudi --max_train 4323 --out_name saudi-matched
"""
import argparse
import os
import random
import statistics
import sys

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "data")
WOS_ROOT = os.path.join(os.path.dirname(HERE), "word_order_study", "data")

# English length band (wider than the word-order study's 15-30).
MIN_WORDS, MAX_WORDS = 3, 80
RATIO_LO_FACTOR, RATIO_HI_FACTOR = 1 / 3, 3.0
SEED = 42

# Same HF sources as word_order_study/source_step1_data.py.
LANGS = {
    "vietnamese": {"kind": "flat", "hf": "thainq107/iwslt2015-en-vi",
                   "en_col": "en", "tgt_col": "vi", "ext": "vi", "order": "SVO"},
    "msa":        {"kind": "flat", "hf": "NeutrinoPit/TED2020-en-ar",
                   "en_col": "en", "tgt_col": "ar", "ext": "ar", "order": "VSO"},
    "korean":     {"kind": "flat", "hf": "msarmi9/korean-english-multitarget-ted-talks-task",
                   "en_col": "english", "tgt_col": "korean", "ext": "ko", "order": "SOV"},
    "saudi":      {"kind": "alexandria", "country": "SA", "ext": "ar", "order": "VSO-leaning"},
    "egyptian":   {"kind": "alexandria", "country": "EG", "ext": "ar", "order": "SVO"},
}


def norm_en(s):
    """Lowercase and collapse whitespace, to use as the dedup and holdout key."""
    return " ".join(s.lower().split())


def in_band(en):
    """True if the English sentence length is inside the band."""
    return MIN_WORDS <= len(en.split()) <= MAX_WORDS


def read_holdout(lang, ext):
    """English keys of the word_order_study dev and test splits, which train must not contain."""
    keys = set()
    for split in ("dev", "test"):
        path = os.path.join(WOS_ROOT, lang, f"{split}.en")
        if not os.path.exists(path):
            print(f"  WARNING: no holdout file {path}")
            continue
        with open(path, encoding="utf-8") as f:
            keys.update(norm_en(line) for line in f if line.strip())
    return keys


# ---------------------------------------------------------------- extraction

def extract_flat(cfg):
    """Load a plain sentence-pair dataset, keeping the pairs where both sides are non-empty."""
    rows = load_dataset(cfg["hf"])["train"]
    en_c, tgt_c = cfg["en_col"], cfg["tgt_col"]
    pairs = []
    for r in rows:
        en = (r[en_c] or "").strip()
        tgt = (r[tgt_c] or "").strip()
        if en and tgt:
            pairs.append({"en": en, "tgt": tgt})
    return pairs


def extract_alexandria(cfg):
    """Flatten every Alexandria conversation for one country into turn-level sentence pairs."""
    ds = load_dataset("UBC-NLP/alexandria", name=cfg["country"])
    pairs = []
    for split in ds:
        for conv in ds[split]:
            for e, d in zip(conv["english_conversation"], conv["dialectal_conversation"]):
                en, tgt = e["text"].strip(), d["text"].strip()
                if en and tgt:
                    pairs.append({"en": en, "tgt": tgt, "domain": conv.get("domain", "_")})
    return pairs


def clean_pool(name, cfg):
    """Filter by length, dedup, drop outlier ratios, remove held-out sentences."""
    raw = extract_flat(cfg) if cfg["kind"] == "flat" else extract_alexandria(cfg)
    band = [p for p in raw if in_band(p["en"])]

    seen, deduped = set(), []
    for p in band:
        k = norm_en(p["en"])
        if k not in seen:
            seen.add(k)
            deduped.append(p)

    ratios = [len(p["tgt"]) / max(1, len(p["en"])) for p in deduped]
    med = statistics.median(ratios) if ratios else 0.0
    lo, hi = med * RATIO_LO_FACTOR, med * RATIO_HI_FACTOR
    clean = [p for p in deduped
             if lo <= len(p["tgt"]) / max(1, len(p["en"])) <= hi]

    holdout = read_holdout(name, cfg["ext"])
    pool = [p for p in clean if norm_en(p["en"]) not in holdout]

    stats = {"raw": len(raw), "band": len(band), "deduped": len(deduped),
             "clean": len(clean), "pool": len(pool), "holdout": len(holdout),
             "ratio_med": round(med, 2)}
    return pool, stats


# ---------------------------------------------------------------- writing

def copy_split(lang, ext, split, out_name=None):
    """Copy one word_order_study split into this study's data dir, unchanged."""
    n = 0
    for side in ("en", ext):
        src = os.path.join(WOS_ROOT, lang, f"{split}.{side}")
        dst = os.path.join(OUT_ROOT, out_name or lang, f"{split}.{side}")
        with open(src, encoding="utf-8") as f:
            text = f.read()
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        n = text.count("\n")
    return n


def build_language(name, cfg, report, max_train=None, out_name=None):
    """Write train from the cleaned pool, and copy dev and test from the word-order study."""
    print(f"\n########## {name} ({cfg['order']}) ##########")
    pool, st = clean_pool(name, cfg)
    print(f"  raw={st['raw']}  in[{MIN_WORDS},{MAX_WORDS}]={st['band']}  deduped={st['deduped']}  "
          f"ratio-med={st['ratio_med']}  clean={st['clean']}  "
          f"minus {st['holdout']} held-out -> train pool={st['pool']}")

    # Sample, not slice (flat sources are ordered by talk).
    st["capped_from"] = len(pool)
    if max_train and len(pool) > max_train:
        random.Random(SEED).shuffle(pool)
        pool = pool[:max_train]
        print(f"  capped to {len(pool)} of {st['capped_from']} (seed {SEED})")

    out_dir = os.path.join(OUT_ROOT, out_name or name)
    os.makedirs(out_dir, exist_ok=True)
    ext = cfg["ext"]
    with open(os.path.join(out_dir, "train.en"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(p["en"] + "\n" for p in pool))
    with open(os.path.join(out_dir, f"train.{ext}"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(p["tgt"] + "\n" for p in pool))
    n_dev = copy_split(name, ext, "dev", out_name)
    n_test = copy_split(name, ext, "test", out_name)
    print(f"  wrote train={len(pool)}  dev={n_dev} (copied)  test={n_test} (copied)")

    st.update({"language": out_name or name, "order": cfg["order"], "train": len(pool),
               "dev": n_dev, "test": n_test})
    report.append(st)


def read_existing_rows(path):
    """Read existing rows from corpus_sizes.md so we merge instead of duplicate."""
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != 10 or cells[0] in ("Dataset", "") or set(cells[1]) <= set("-:"):
                continue
            rows[cells[0]] = line
    return rows


def write_stats(report):
    """Write the corpus-size table that RESULTS.md quotes."""
    lines = ["# ITST corpus sizes",
             "",
             f"Built by `source_itst_data.py`. English length band {MIN_WORDS}-{MAX_WORDS} words, "
             "deduped on English, target/source character ratio kept within 1/3x to 3x of the "
             "language median. Dev and test are copied unchanged from `word_order_study/data/` and "
             "their English sentences are excluded from train.",
             "",
             "| Dataset | Order | Raw pairs | In band | Deduped | Ratio-clean | Pool | Train | "
             "Dev | Test |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, "corpus_sizes.md")

    merged = read_existing_rows(path)
    for r in report:
        merged[r["language"]] = (
            f"| {r['language']} | {r['order']} | {r['raw']} | {r['band']} | "
            f"{r['deduped']} | {r['clean']} | {r['capped_from']} | {r['train']} | "
            f"{r['dev']} | {r['test']} |")
    order = ["vietnamese", "msa", "korean", "saudi", "saudi-matched", "egyptian"]
    lines += [merged[k] for k in order if k in merged]
    lines += [merged[k] for k in merged if k not in order]

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", default=list(LANGS), choices=list(LANGS))
    ap.add_argument("--report_pool", action="store_true",
                    help="only print how many pairs each source gives, write nothing")
    ap.add_argument("--max_train", type=int, default=None,
                    help="cap every language to this many training pairs, sampled with SEED")
    ap.add_argument("--out_name", default=None,
                    help="write under this dataset name instead of the language name")
    args = ap.parse_args()

    if args.out_name and len(args.languages) > 1:
        sys.exit("--out_name names one output dir, so pass exactly one language")

    report = []
    for name in args.languages:
        if args.report_pool:
            pool, st = clean_pool(name, LANGS[name])
            print(f"{name:12s} raw={st['raw']:>7} band={st['band']:>7} dedup={st['deduped']:>7} "
                  f"clean={st['clean']:>7} train_pool={st['pool']:>7}")
        else:
            build_language(name, LANGS[name], report, args.max_train, args.out_name)
    if report:
        write_stats(report)
