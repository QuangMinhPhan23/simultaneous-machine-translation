"""
DPO Stage 4 pairs: the human reference (chosen) vs. a perturbed copy of it (rejected).
Four perturbation types: word_drop (drop 15% of tokens), mlm_replace (replace ~15% using
CAMeLBERT), msa_injection (dialectal function words -> MSA) and dialect_swap (re-render with
the other dialect's markers). Up to one negative per type per sentence, so up to 4x pairs.
"""
import argparse
import json
import os
import random

from transformers import pipeline

from dialect_lexicon import inject_msa, swap_dialect

MLM_CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"


def word_drop(text, rate, rng):
    words = text.split()
    if len(words) < 2:
        return text
    kept = [w for w in words if rng.random() >= rate]
    return " ".join(kept) if kept else text


def mlm_replace(text, rate, rng, fill_mask):
    words = text.split()
    if len(words) < 2:
        return text
    n_replace = max(1, int(len(words) * rate))
    idxs = rng.sample(range(len(words)), min(n_replace, len(words)))
    for idx in idxs:
        masked_words = list(words)
        masked_words[idx] = fill_mask.tokenizer.mask_token
        masked_text = " ".join(masked_words)
        try:
            prediction = fill_mask(masked_text, top_k=1)[0]["token_str"].strip()
        except Exception:
            continue
        if prediction:
            words[idx] = prediction
    return " ".join(words)


def build_perturbations(text, dialect, rng, fill_mask):
    out = {"word_drop": word_drop(text, 0.15, rng)}
    out["mlm_replace"] = mlm_replace(text, 0.15, rng, fill_mask) if fill_mask else None

    try:
        msa_text, n_hits = inject_msa(text, dialect)
        out["msa_injection"] = msa_text if n_hits > 0 else None
    except ValueError:
        out["msa_injection"] = None

    other_dialect = "LB" if dialect == "EG" else "EG"
    try:
        swapped, n_hits = swap_dialect(text, dialect, other_dialect)
        out["dialect_swap"] = swapped if n_hits > 0 else None
    except ValueError:
        out["dialect_swap"] = None

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/mt_data/train_data/Arabic-EG-SiMT-OMT.json")
    parser.add_argument("--dialect", default="EG")
    parser.add_argument("--output", default="data/mt_data/dpo_data/stage4_pairs.json")
    parser.add_argument("--skip_mlm", action="store_true",
                         help="Skip the mlm_replace perturbation, so CAMeLBERT is not downloaded")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    fill_mask = None
    if not args.skip_mlm:
        fill_mask = pipeline("fill-mask", model=MLM_CHECKPOINT)

    with open(args.data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    examples = [ex for ex in examples if "latency" not in ex]  # OMT rows only

    pairs = []
    for ex in examples:
        chosen = ex["output"].strip()
        prompt = f"{ex['instruction']}\n{ex['input']}" if ex.get("input") else ex["instruction"]
        perturbations = build_perturbations(chosen, args.dialect, rng, fill_mask)
        for kind, rejected in perturbations.items():
            if not rejected or rejected.strip() == chosen:
                continue
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected.strip(),
                "perturbation": kind,
            })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    by_kind = {}
    for p in pairs:
        by_kind[p["perturbation"]] = by_kind.get(p["perturbation"], 0) + 1
    print(f"Wrote {len(pairs)} Stage 4 pairs to {args.output}: {by_kind}")
