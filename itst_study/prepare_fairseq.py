"""Encode and binarize parallel text for fairseq.

--mode paper: IWSLT15 En-Vi with word-level vocab (paper reproduction).
--mode study: study languages with joint SentencePiece 10k BPE.
Dialects reuse MSA's SPM model and dictionary (--spm_from msa).

Usage:
  python itst_study/prepare_fairseq.py --mode paper
  python itst_study/prepare_fairseq.py --mode study --languages vietnamese msa korean
  python itst_study/prepare_fairseq.py --mode study --languages saudi egyptian --spm_from msa
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BIN = os.path.join(HERE, "data-bin")
SPM = os.path.join(HERE, "spm")

EXT = {"vietnamese": "vi", "msa": "ar", "korean": "ko", "saudi": "ar", "egyptian": "ar",
       "saudi-matched": "ar"}

# HF mirror of IWSLT15, pre-tokenized.
IWSLT_HF = "thainq107/iwslt2015-en-vi"
DEV_SIZE = 1553  # held out from train as dev (tst2012 not available in this mirror)

VOCAB_SIZE = 10000


def run(cmd, **kw):
    """Run a command, showing it first, and stop the script if it fails."""
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


# ---------------------------------------------------------------- paper track

def write_pairs(path_prefix, pairs):
    """Write a list of (en, vi) pairs as two line-aligned files."""
    for side, idx in (("en", 0), ("vi", 1)):
        with open(f"{path_prefix}.{side}", "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(p[idx].replace("\n", " ").strip() + "\n" for p in pairs))


def fetch_iwslt(dest):
    """Write train/dev/test for the paper track from the Hugging Face IWSLT15 mirror."""
    os.makedirs(dest, exist_ok=True)
    if all(os.path.exists(os.path.join(dest, f"{s}.{x}"))
           for s in ("train", "dev", "test") for x in ("en", "vi")):
        return

    # Lazy import (not available in the fairseq env).
    from datasets import load_dataset
    ds = load_dataset(IWSLT_HF)
    all_train = [(r["en"], r["vi"]) for r in ds["train"]
                 if (r["en"] or "").strip() and (r["vi"] or "").strip()]
    test = [(r["en"], r["vi"]) for r in ds["test"]
            if (r["en"] or "").strip() and (r["vi"] or "").strip()]

    # Hold the tail of train out as dev, since this mirror has no tst2012.
    train, dev = all_train[:-DEV_SIZE], all_train[-DEV_SIZE:]

    # The test set must not appear in train, or the reproduction number is meaningless.
    train_keys = {" ".join(e.lower().split()) for e, _ in train}
    leaked = sum(1 for e, _ in test if " ".join(e.lower().split()) in train_keys)
    print(f"  train={len(train)}  dev={len(dev)}  test={len(test)}  "
          f"test sentences also in train: {leaked}")

    write_pairs(os.path.join(dest, "train"), train)
    write_pairs(os.path.join(dest, "dev"), dev)
    write_pairs(os.path.join(dest, "test"), test)


def prepare_paper(text_only=False):
    """Build data-bin/paper-envi with the paper's word-level vocabulary."""
    raw = os.path.join(DATA, "paper-envi")
    fetch_iwslt(raw)
    for s in ("train", "dev", "test"):
        print(f"  {s}: {count_lines(os.path.join(raw, s + '.en'))} pairs")
    if text_only:
        print(f"OK: text written to {raw}")
        return

    out = os.path.join(BIN, "paper-envi")
    shutil.rmtree(out, ignore_errors=True)
    # Frequency threshold 5 on both sides is what the paper describes; fairseq maps everything
    # below it to <unk>. No joined dictionary here, the two vocabularies are separate.
    run(["fairseq-preprocess", "--source-lang", "en", "--target-lang", "vi",
         "--trainpref", os.path.join(raw, "train"),
         "--validpref", os.path.join(raw, "dev"),
         "--testpref", os.path.join(raw, "test"),
         "--thresholdsrc", "5", "--thresholdtgt", "5",
         "--destdir", out, "--workers", "8"])
    print(f"OK: {out}")


# ---------------------------------------------------------------- study track

def train_spm(lang, ext, model_prefix):
    """Learn one joint SentencePiece model over both sides of this language's training data."""
    import sentencepiece as spm
    src = os.path.join(DATA, lang, "train.en")
    tgt = os.path.join(DATA, lang, f"train.{ext}")
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=f"{src},{tgt}",
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        input_sentence_size=2000000,
        shuffle_input_sentence=True,
    )


def encode_spm(model_path, in_path, out_path):
    """Write in_path through the SentencePiece model, one space-joined piece sequence per line."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_path)
    with open(in_path, encoding="utf-8") as fi, \
         open(out_path, "w", encoding="utf-8", newline="\n") as fo:
        for line in fi:
            fo.write(" ".join(sp.encode(line.strip(), out_type=str)) + "\n")


def prepare_study(lang, spm_from=None):
    """Encode one language with SentencePiece and binarize it for fairseq."""
    ext = EXT[lang]
    print(f"\n########## {lang} (en-{ext}) ##########")
    for split in ("train", "dev", "test"):
        for side in ("en", ext):
            p = os.path.join(DATA, lang, f"{split}.{side}")
            if not os.path.exists(p):
                sys.exit(f"missing {p} - run source_itst_data.py first")
        print(f"  {split}: {count_lines(os.path.join(DATA, lang, f'{split}.en'))} pairs")

    # Step 1: get the SentencePiece model, either learn it or borrow another language's.
    owner = spm_from or lang
    model_prefix = os.path.join(SPM, owner, "spm")
    model_path = model_prefix + ".model"
    if not os.path.exists(model_path):
        if spm_from:
            sys.exit(f"missing {model_path} - prepare {spm_from} first")
        print(f"  training SentencePiece ({VOCAB_SIZE} joint BPE)")
        train_spm(lang, ext, model_prefix)
    else:
        print(f"  reusing SentencePiece model {model_path}")

    # Step 2: encode every split.
    enc_dir = os.path.join(DATA, lang, "spm")
    os.makedirs(enc_dir, exist_ok=True)
    for split in ("train", "dev", "test"):
        for side in ("en", ext):
            encode_spm(model_path,
                       os.path.join(DATA, lang, f"{split}.{side}"),
                       os.path.join(enc_dir, f"{split}.{side}"))

    # Step 3: binarize. A joined dictionary is required by --share-all-embeddings at training time.
    # A dialect borrows MSA's dictionary too, so its checkpoint can start from the MSA one.
    out = os.path.join(BIN, lang)
    shutil.rmtree(out, ignore_errors=True)
    cmd = ["fairseq-preprocess", "--source-lang", "en", "--target-lang", ext,
           "--trainpref", os.path.join(enc_dir, "train"),
           "--validpref", os.path.join(enc_dir, "dev"),
           "--testpref", os.path.join(enc_dir, "test"),
           "--joined-dictionary", "--destdir", out, "--workers", "8"]
    if spm_from:
        shared = os.path.join(BIN, spm_from, f"dict.{ext}.txt")
        if not os.path.exists(shared):
            sys.exit(f"missing {shared} - prepare {spm_from} first")
        cmd += ["--srcdict", shared]
    run(cmd)
    print(f"OK: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "study"], required=True)
    ap.add_argument("--languages", nargs="+", default=["vietnamese", "msa", "korean"],
                    choices=list(EXT))
    ap.add_argument("--spm_from", choices=list(EXT), default=None,
                    help="reuse this language's SentencePiece model and dictionary")
    ap.add_argument("--text_only", action="store_true",
                    help="write the text files and stop, for running where fairseq is not installed")
    args = ap.parse_args()

    os.makedirs(BIN, exist_ok=True)
    if args.mode == "paper":
        prepare_paper(text_only=args.text_only)
    else:
        for lang in args.languages:
            prepare_study(lang, spm_from=args.spm_from)
