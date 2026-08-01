"""Step 1: data sourcing for the word-order study.

Downloads the source dataset of each language (see LANGS below) and writes six line-aligned
parallel files per language (train/dev/test x .en/.<tgt>) plus a stats table under data/.
Cleaning: keep English sentences of 15-30 words, dedup on English, drop pairs whose
target/source character ratio is far from the language median, then sample the split sizes.

Usage:
  python word_order_study/source_step1_data.py [--languages vietnamese saudi]
"""
import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "data")

# English length band, and how far the target/source char ratio may sit from the median.
MIN_WORDS, MAX_WORDS = 15, 30
RATIO_LO_FACTOR, RATIO_HI_FACTOR = 1 / 3, 3.0

# Window size for MATTR, in target tokens. Must be <= the smallest split's token count.
MATTR_WINDOW = 100

# (train, dev, test) targets
FLAT_TARGETS = (2400, 300, 300)
ALEX_TARGETS = (3000, 500, 500)

# Per-language config. "kind" picks the reader: "flat" is one pool of sentence pairs,
# "alexandria" rows are conversations that already have splits. "ext" is the target file suffix.
LANGS = {
    "vietnamese": {"kind": "flat", "hf": "thainq107/iwslt2015-en-vi",
                   "en_col": "en", "tgt_col": "vi", "ext": "vi",
                   "order": "SVO", "targets": FLAT_TARGETS},
    "msa":        {"kind": "flat", "hf": "NeutrinoPit/TED2020-en-ar",
                   "en_col": "en", "tgt_col": "ar", "ext": "ar",
                   "order": "VSO", "targets": FLAT_TARGETS},
    "korean":     {"kind": "flat", "hf": "msarmi9/korean-english-multitarget-ted-talks-task",
                   "en_col": "english", "tgt_col": "korean", "ext": "ko",
                   "order": "SOV", "targets": FLAT_TARGETS},
    "saudi":      {"kind": "alexandria", "country": "SA", "ext": "ar",
                   "order": "VSO-leaning", "targets": ALEX_TARGETS},
    "egyptian":   {"kind": "alexandria", "country": "EG", "ext": "ar",
                   "order": "SVO", "targets": ALEX_TARGETS},
}


def norm_en(s):
    """Lowercase and collapse whitespace, to use as the dedup key."""
    return " ".join(s.lower().split())


def n_words(s):
    return len(s.split())


def in_band(en):
    """True if the English sentence is inside the 15-30 word length band."""
    return MIN_WORDS <= n_words(en) <= MAX_WORDS


def dedup(pairs, seen=None):
    """Dedup on normalized English. Pass `seen` to also exclude keys used somewhere else."""
    if seen is None:
        seen = set()
    out = []
    for p in pairs:
        k = norm_en(p["en"])
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out, seen


def ratio_bounds(pairs):
    """Return (lo, hi, median) of the target/source character ratio for this language."""
    ratios = [len(p["tgt"]) / max(1, len(p["en"])) for p in pairs]
    med = statistics.median(ratios)
    return med * RATIO_LO_FACTOR, med * RATIO_HI_FACTOR, med


def apply_ratio(pairs, lo, hi):
    """Keep only the pairs whose target/source character ratio falls inside [lo, hi]."""
    return [p for p in pairs if lo <= len(p["tgt"]) / max(1, len(p["en"])) <= hi]


def stratified(items, n, rng):
    """Sample n items, as evenly spread across item['domain'] as the data allows."""
    # Shuffle inside each domain, then take an equal share from every domain.
    buckets = defaultdict(list)
    for it in items:
        buckets[it.get("domain", "_")].append(it)
    for b in buckets.values():
        rng.shuffle(b)
    keys = list(buckets)
    base = n // len(keys)
    chosen, leftover = [], []
    for k in keys:
        take = min(base, len(buckets[k]))
        chosen += buckets[k][:take]
        leftover += buckets[k][take:]
    # Small domains run out early, so top up from what is left.
    rng.shuffle(leftover)
    chosen += leftover[: max(0, n - len(chosen))]
    rng.shuffle(chosen)
    return chosen[:n]


# ---------------------------------------------------------------- extraction

def extract_flat(cfg):
    """Load a plain sentence-pair dataset from the Hub, keeping pairs where both sides exist."""
    ds = load_dataset(cfg["hf"])
    rows = ds["train"]
    en_c, tgt_c = cfg["en_col"], cfg["tgt_col"]
    pairs = []
    for r in rows:
        en = (r[en_c] or "").strip()
        tgt = (r[tgt_c] or "").strip()
        if en and tgt:
            pairs.append({"en": en, "tgt": tgt})
    return {"train": pairs}  # one pool, cut into splits later


def extract_alexandria(cfg):
    """Flatten the Alexandria dialect data for one country into sentence pairs.

    Each row is a conversation, so the English and dialectal turns are zipped turn by turn."""
    ds = load_dataset("UBC-NLP/alexandria", name=cfg["country"])
    out = {}
    split_map = {"train": "train", "dev": "dev", "test": "test"}
    for logical, native in split_map.items():
        pairs = []
        for conv in ds[native]:
            for e, d in zip(conv["english_conversation"], conv["dialectal_conversation"]):
                en, tgt = e["text"].strip(), d["text"].strip()
                if en and tgt:
                    pairs.append({"en": en, "tgt": tgt, "domain": conv["domain"]})
        out[logical] = pairs
    return out


# ---------------------------------------------------------------- per-language build

def build_language(name, cfg, rng, report):
    """Clean one language, write its six parallel files, and append its stats rows to `report`."""
    print(f"\n########## {name} ({cfg['order']}) ##########")
    t_tr, t_dev, t_te = cfg["targets"]

    if cfg["kind"] == "flat":
        # Step 1a (flat datasets): clean the single pool, then cut three disjoint slices from it.
        raw = extract_flat(cfg)["train"]
        band = [p for p in raw if in_band(p["en"])]
        deduped, _ = dedup(band)
        lo, hi, med = ratio_bounds(deduped)
        clean = apply_ratio(deduped, lo, hi)
        rng.shuffle(clean)
        need = t_tr + t_dev + t_te
        if len(clean) < need:
            print(f"  WARNING: only {len(clean)} clean pairs < {need} needed")
        splits = {
            "train": clean[:t_tr],
            "dev": clean[t_tr:t_tr + t_dev],
            "test": clean[t_tr + t_dev:t_tr + t_dev + t_te],
        }
        print(f"  raw={len(raw)}  in[15,30]={len(band)}  deduped={len(deduped)}  "
              f"ratio-med={med:.2f} keep=[{lo:.2f},{hi:.2f}]  clean={len(clean)}")
    else:  # alexandria
        # Step 1b (Alexandria): the splits already exist, so clean train first, then reuse its
        # ratio window and its English keys when cleaning dev and test.
        raw = extract_alexandria(cfg)
        tr_band = [p for p in raw["train"] if in_band(p["en"])]
        tr_dd, seen = dedup(tr_band)
        lo, hi, med = ratio_bounds(tr_dd)
        tr_clean = apply_ratio(tr_dd, lo, hi)

        def prep_eval(split):
            band = [p for p in raw[split] if in_band(p["en"])]
            dd, _ = dedup(band, seen=set(seen))  # drop anything already in train
            return apply_ratio(dd, lo, hi)

        dev_clean = prep_eval("dev")
        te_clean = prep_eval("test")
        splits = {
            "train": stratified(tr_clean, t_tr, rng),
            "dev": stratified(dev_clean, t_dev, rng),
            "test": stratified(te_clean, t_te, rng),
        }
        for s, avail in [("train", len(tr_clean)), ("dev", len(dev_clean)), ("test", len(te_clean))]:
            if len(splits[s]) < cfg["targets"][["train", "dev", "test"].index(s)]:
                print(f"  NOTE: {s} capped at {len(splits[s])} (only {avail} clean available)")
        print(f"  train raw={len(raw['train'])} clean={len(tr_clean)} | "
              f"dev clean={len(dev_clean)} | test clean={len(te_clean)}  "
              f"ratio-med={med:.2f} keep=[{lo:.2f},{hi:.2f}]")

    # Step 2: write one file per split and side, keeping the two sides line-aligned.
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    ext = cfg["ext"]
    for split, pairs in splits.items():
        with open(os.path.join(out_dir, f"{split}.en"), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(p["en"] + "\n" for p in pairs))
        with open(os.path.join(out_dir, f"{split}.{ext}"), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(p["tgt"] + "\n" for p in pairs))
        stats = split_stats(name, cfg, split, pairs)
        report.append(stats)
        print(f"  wrote {split}: {stats['n']:>4} pairs -> {split}.en / {split}.{ext}")
    return splits


def mattr(tokens, window=MATTR_WINDOW):
    """Moving-average type-token ratio: mean TTR over sliding windows of `window` tokens.

    Unlike plain TTR it does not shrink as the corpus grows, so splits of different size compare.
    Streams shorter than one window get plain TTR."""
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    # Slide the window one token at a time, adding each window's ratio to a running sum.
    counts = Counter(tokens[:window])
    ttr_sum = len(counts) / window
    for i in range(window, n):
        counts[tokens[i]] += 1
        old = tokens[i - window]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        ttr_sum += len(counts) / window
    return ttr_sum / (n - window + 1)


def split_stats(name, cfg, split, pairs):
    """One stats row for a split: pair count, average lengths, and target TTR and MATTR."""
    if not pairs:
        return {"language": name, "order": cfg["order"], "split": split, "n": 0}
    src_w = [n_words(p["en"]) for p in pairs]
    tgt_w = [n_words(p["tgt"]) for p in pairs]
    src_c = [len(p["en"]) for p in pairs]
    tgt_c = [len(p["tgt"]) for p in pairs]
    tgt_toks = []
    for p in pairs:
        tgt_toks.extend(p["tgt"].lower().split())
    tgt_tokens = len(tgt_toks)
    tgt_types = len(set(tgt_toks))
    return {
        "language": name, "order": cfg["order"], "split": split, "n": len(pairs),
        "avg_src_words": round(statistics.mean(src_w), 2),
        "avg_tgt_words": round(statistics.mean(tgt_w), 2),
        "avg_src_chars": round(statistics.mean(src_c), 1),
        "avg_tgt_chars": round(statistics.mean(tgt_c), 1),
        "ttr_tgt": round(tgt_types / max(1, tgt_tokens), 4),
        "mattr_tgt": round(mattr(tgt_toks), 4),
        "mattr_window": MATTR_WINDOW,
        "tgt_tokens": tgt_tokens,
    }


def write_stats(report):
    """Write the stats rows twice: stats.json for later scripts, stats.md to read."""
    lines = ["# Step 1 - Data Sourcing Statistics",
             "",
             "English source filtered to 15-30 whitespace words; deduped on English; robust "
             "target/source char-ratio misalignment filter.",
             "",
             "**Lexical diversity.** `TTR` = target types / target tokens (whitespace-split, "
             "lowercased) over the whole split. Raw TTR falls as a corpus grows (tokens get "
             f"reused), so it is *not* comparable across the differently-sized splits; `MATTR` "
             f"(mean TTR over sliding windows of {MATTR_WINDOW} target tokens, Covington & McFall "
             "2010) removes that size dependence and is the column to compare across splits. "
             "**Both** still reflect segmentation/morphology, which differ by script "
             "(Vietnamese=syllables, Korean=eojeol, Arabic=words), so *cross-language* values "
             "measure tokenization + morphological richness as much as diversity - read them within "
             "a language, not as a cross-language ranking.",
             "",
             "| Language | Order | Split | Pairs | Avg src words | Avg tgt words | Avg src chars | "
             f"Avg tgt chars | TTR (tgt) | MATTR (tgt, W={MATTR_WINDOW}) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report:
        if not r.get("n"):
            continue
        lines.append(f"| {r['language']} | {r['order']} | {r['split']} | {r['n']} | "
                     f"{r['avg_src_words']} | {r['avg_tgt_words']} | {r['avg_src_chars']} | "
                     f"{r['avg_tgt_chars']} | {r['ttr_tgt']} | {r.get('mattr_tgt', '')} |")
    with open(os.path.join(OUT_ROOT, "stats.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_ROOT}/stats.md and stats.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", default=list(LANGS), choices=list(LANGS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # One seeded generator for every language, so re-running gives the same splits.
    rng = random.Random(args.seed)
    os.makedirs(OUT_ROOT, exist_ok=True)
    report = []
    for name in args.languages:
        build_language(name, LANGS[name], rng, report)
    write_stats(report)
    print("\nStep 1 complete.")
