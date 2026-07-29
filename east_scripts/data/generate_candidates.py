"""
Generic OMT candidate generator shared by the DPO stage builders. Reads an alpaca-schema
file and writes {"instruction", "input", "reference", "prediction"} rows, with no scoring.

Always run it against the checkpoint the next DPO stage will train from (SFT_MODEL for
Stage 1, the Stage-2 checkpoint for Stage 3). Negatives from a much older checkpoint make
DPO training unstable.
"""
import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True,
                         help="Alpaca-schema file; SiMT rows (those with a 'latency' key) are "
                              "filtered out automatically")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Chat-tuned checkpoints often list only the base EOS, not the turn-end token.
    eos_token_ids = {tokenizer.eos_token_id}
    for turn_end_token in ("<|eot_id|>", "<end_of_turn>", "<|im_end|>"):
        turn_end_id = tokenizer.convert_tokens_to_ids(turn_end_token)
        if turn_end_id is not None and turn_end_id != tokenizer.unk_token_id:
            eos_token_ids.add(turn_end_id)
    eos_token_ids = list(eos_token_ids)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    with open(args.data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    examples = [ex for ex in examples if "latency" not in ex]
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(f"Generating candidates for {len(examples)} OMT examples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    with torch.no_grad():
        for ex in tqdm(examples):
            user_content = ex["instruction"]
            if ex.get("input"):
                user_content = f"{user_content}\n{ex['input']}"
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(device)
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
            )
            generated = output_ids[0][inputs.input_ids.shape[1]:]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

            results.append({
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "reference": ex["output"],
                "prediction": prediction,
                "src_lang": ex.get("src_lang"),
                "tgt_lang": ex.get("tgt_lang"),
            })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} candidates to {args.output}")


if __name__ == "__main__":
    main()
