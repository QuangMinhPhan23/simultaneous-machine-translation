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
    # The lang-code lookup API changed across transformers versions, so try both.
    if hasattr(tokenizer, "lang_code_to_id"):
        return tokenizer.lang_code_to_id[NLLB_MSA_CODE]
    return tokenizer.convert_tokens_to_ids(NLLB_MSA_CODE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                         help="Alexandria test-data file with a 'source' field. Must be the same "
                              "unfiltered file used for the predictions, as rescoring joins by "
                              "list position.")
    parser.add_argument("--model_name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    forced_bos_token_id = get_msa_forced_bos_id(tokenizer)

    with open(args.data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    with torch.no_grad():
        for idx, ex in enumerate(tqdm(examples)):
            inputs = tokenizer(ex["source"], return_tensors="pt").to(device)
            output_ids = model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=args.max_new_tokens,
            )
            msa_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            results.append({"index": idx, "source": ex["source"], "msa_reference": msa_text})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} MSA pseudo-references to {args.output}")


if __name__ == "__main__":
    main()
