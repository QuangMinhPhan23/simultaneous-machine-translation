"""
Plain offline (non-SiMT) translation eval, for comparing how different base
models handle Egyptian Arabic. No read/write tokens are used.

Input: a test JSON file and a model path.
Output: predictions.json and results.json (BLEU, and COMET if a checkpoint is given).
Not every Arabic model ships a chat template, so smoke-test --prompt_style
with --max_examples 3 before a full run.
"""
import argparse
import json
import os
from statistics import mean

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sacrebleu

INSTRUCTION = "You are a professional translator, your task is to translate the following text from {src_lang} into {tgt_lang}."
ACEGPT_AR_SYSTEM = "أنت مساعد مفيد ومحترم وصادق."


def build_prompt(tokenizer, prompt_style, src_lang, tgt_lang, source):
    instruction = INSTRUCTION.format(src_lang=src_lang, tgt_lang=tgt_lang)

    if prompt_style == "chat_template":
        messages = [{"role": "user", "content": f"{instruction}\n{source}"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if prompt_style == "llama3_manual":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}\n{source}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    if prompt_style == "llama2_inst":
        return f"[INST] <<SYS>>\n{ACEGPT_AR_SYSTEM}\n<</SYS>>\n\n{instruction}\n{source} [/INST]"

    if prompt_style == "alpaca":
        return f"### Instruction:\n{instruction}\n\n### Input:\n{source}\n\n### Response:\n"

    raise ValueError(f"Unknown prompt_style: {prompt_style}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--prompt_style", required=True,
        choices=["chat_template", "llama3_manual", "llama2_inst", "alpaca"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_examples", type=int, default=None,
                         help="Cap test set size; use e.g. 3 for a quick template smoke test")
    parser.add_argument("--comet_ckpt_path", type=str, default=None,
                         help="Optional; omit to skip COMET")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Many chat models list only the base EOS, so add turn-end tokens or
    # generation keeps going and invents extra turns.
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
        data = json.load(f)
    if args.max_examples:
        data = data[: args.max_examples]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = []

    with torch.no_grad():
        for sample in tqdm(data):
            prompt = build_prompt(tokenizer, args.prompt_style, sample["src_lang"], sample["tgt_lang"], sample["source"])
            inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(device)
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=5,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
            )
            generated = output_ids[0][inputs.input_ids.shape[1]:]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

            predictions.append(
                {
                    "source": sample["source"],
                    "reference": sample["reference"],
                    "prediction": prediction,
                    "prompt": prompt,
                }
            )

    hypos = [p["prediction"] for p in predictions]
    refs = [p["reference"] for p in predictions]
    bleu = sacrebleu.corpus_bleu(hypos, [refs], tokenize="13a").score

    results = {"BLEU": bleu, "prompt_style": args.prompt_style, "n_examples": len(predictions)}

    os.makedirs(args.output_dir, exist_ok=True)

    def save():
        with open(os.path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Save before COMET so a scoring crash never throws away the generation.
    save()

    if args.comet_ckpt_path:
        try:
            from comet import load_from_checkpoint
            comet_model = load_from_checkpoint(args.comet_ckpt_path, reload_hparams=True)
            comet_data = [{"src": p["source"], "mt": p["prediction"], "ref": p["reference"]} for p in predictions]
            gpus = 1 if torch.cuda.is_available() else 0
            comet_output = comet_model.predict(comet_data, batch_size=256, gpus=gpus)
            for i, score in enumerate(comet_output.scores):
                predictions[i]["COMET"] = score * 100
            results["COMET"] = comet_output.system_score * 100
            save()
        except Exception as e:
            print(f"WARNING: COMET scoring failed ({e}); BLEU + predictions were already saved.")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    for p in predictions[:3]:
        print("SRC:", p["source"])
        print("REF:", p["reference"])
        print("HYP:", p["prediction"])
        print("---")


if __name__ == "__main__":
    main()
