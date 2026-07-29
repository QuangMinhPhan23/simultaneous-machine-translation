"""
Generate a pseudo-MSA reference for an Alexandria test-data file, so existing dialect
predictions can be rescored against MSA without generating them again.

Alexandria has no MSA field of its own, so the reference is translated from the English
source with NLLB-200. It is a directional check on MSA leakage, not ground truth.
"""
import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_MSA_CODE = "arb_Arab"  # FLORES-200 code for Modern Standard Arabic


def get_msa_forced_bos_id(tokenizer):
    """Returns the token id NLLB uses to force its output into MSA."""
    # The lang-code lookup API changed across transformers versions, so try both.
    if hasattr(tokenizer, "lang_code_to_id"):
        return tokenizer.lang_code_to_id[NLLB_MSA_CODE]
    return tokenizer.convert_tokens_to_ids(NLLB_MSA_CODE)


def main():
    """Translates the English source of every test example into MSA and writes the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                         help="Alexandria test-data file with a 'source' field. Must be the same "
                              "unfiltered file used for the predictions, as rescoring joins by "
                              "list position.")
    parser.add_argument("--model_name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    # Step 1: load NLLB with English set as the source language.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # NLLB picks its output language from the first decoder token, which this id pins to MSA.
    forced_bos_token_id = get_msa_forced_bos_id(tokenizer)

    with open(args.data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    # Step 2: translate each English source into MSA, keeping the original list order.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    with torch.no_grad():
        for idx, ex in enumerate(tqdm(examples)):
            inputs = tokenizer(ex["source"], return_tensors="pt").to(device)
            output_ids = model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=args.max_new_tokens,
            )
            # One sentence per call, so batch_decode returns a one-item list to unpack.
            msa_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # idx is the position in the input file, which is the key rescoring joins on.
            results.append({"index": idx, "source": ex["source"], "msa_reference": msa_text})

    # Step 3: save the MSA text along with its position, so rescoring can line the two files up.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} MSA pseudo-references to {args.output}")


if __name__ == "__main__":
    main()
