"""Stage II LoRA fine-tuning: adapt a Stage-I SiMT model to a new language.

Set --eot_token to match the base model: "<|eot_id|>" for EAST-8B (default),
"<end_of_turn>" for Nile-Chat. Use --adapter_only to save just the LoRA
adapter instead of the merged model. Set --hub_repo_id (or $HF_REPO_ID)
together with $HF_TOKEN to push the result to the HF Hub.

  python finetune_lora_stage2.py --model_path biaofu-xmu/EAST-8B \
      --data_path <train.json> --output_dir <dir>
"""
import argparse
import json
import os
import random
import sys

import torch
try:
    # peft does an isinstance(..., DTensor) check that only resolves if this
    # submodule has been imported at least once.
    import torch.distributed.tensor  # noqa: F401
except ImportError:
    pass
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class AlpacaSFTDataset(Dataset):
    def __init__(self, examples, tokenizer, cutoff_len, eot_token):
        self.examples = examples
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.eot_token = eot_token

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        user_content = ex["instruction"]
        if ex.get("input"):
            user_content = f"{user_content}\n{ex['input']}"

        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = f"{prompt_text}{ex['output']}{self.eot_token}"

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids
        full_ids = full_ids[: self.cutoff_len]

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = -100

        return {"input_ids": full_ids, "labels": labels}


def make_collate_fn(pad_token_id):
    def collate(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    return collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                         help="A Stage-I checkpoint, e.g. biaofu-xmu/EAST-8B")
    parser.add_argument("--eot_token", default="<|eot_id|>",
                         help="Turn-end token: '<|eot_id|>' for Llama-3/EAST-8B (default), "
                              "'<end_of_turn>' for Gemma-3/Nile-Chat")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=50,
                         help="Checkpoint every N steps so a killed job can resume")
    parser.add_argument("--max_examples", type=int, default=None,
                         help="Cap dataset size; use a small number for a smoke test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub_repo_id", default=os.environ.get("HF_REPO_ID"),
                         help="If set (or $HF_REPO_ID), push checkpoints to this HF Hub repo. "
                              "Requires $HF_TOKEN to be exported.")
    parser.add_argument("--adapter_only", action="store_true",
                         help="Save/push only the LoRA adapter instead of the merged full model; "
                              "eval then loads base + adapter. Saves a lot of Hub space.")
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    tokenizer.add_special_tokens({"eos_token": args.eot_token})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    with open(args.data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    random.shuffle(examples)
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(f"Training on {len(examples)} examples")

    dataset = AlpacaSFTDataset(examples, tokenizer, args.cutoff_len, args.eot_token)

    push_to_hub = bool(args.hub_repo_id)
    if push_to_hub and not os.environ.get("HF_TOKEN"):
        print("WARNING: --hub_repo_id/HF_REPO_ID set but $HF_TOKEN is not -- disabling auto-push.", file=sys.stderr)
        push_to_hub = False
    # Adapter-only mode pushes one small adapter at the end, so the Trainer's own pushes are off.
    trainer_push = push_to_hub and not args.adapter_only

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        push_to_hub=trainer_push,
        hub_model_id=args.hub_repo_id if trainer_push else None,
        # "checkpoint" keeps only the latest one on the Hub instead of every one, to save quota.
        hub_strategy="checkpoint" if trainer_push else "every_save",
        hub_private_repo=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
    )

    last_checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if args.adapter_only:
        adapter_dir = f"{args.output_dir.rstrip('/')}/adapter"
        print(f"Saving LoRA adapter (not merged) to {adapter_dir}")
        model.save_pretrained(adapter_dir)          # peft writes only the adapter weights + config
        tokenizer.save_pretrained(adapter_dir)
        if push_to_hub:
            print(f"Pushing LoRA adapter to https://huggingface.co/{args.hub_repo_id}")
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_repo(repo_id=args.hub_repo_id, private=True, exist_ok=True)
                api.upload_folder(repo_id=args.hub_repo_id, folder_path=adapter_dir,
                                  commit_message="Stage II: LoRA adapter (adapter-only)")
            except Exception as e:
                print(f"WARNING: adapter push to {args.hub_repo_id} failed ({e}); local adapter at "
                      f"{adapter_dir} is intact.", file=sys.stderr)
        print("Done.")
        return

    merged_dir = f"{args.output_dir.rstrip('/')}/merged"
    print(f"Merging LoRA adapter and saving to {merged_dir}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    if push_to_hub:
        # The Trainer only pushed unmerged adapters, so send the merged model as the last commit.
        print(f"Pushing final merged checkpoint to https://huggingface.co/{args.hub_repo_id}")
        try:
            from huggingface_hub import HfApi
            HfApi().upload_folder(repo_id=args.hub_repo_id, folder_path=merged_dir, commit_message="Stage II: merged LoRA checkpoint")
        except Exception as e:
            # Warn instead of raising, so a push failure does not kill the eval steps after this.
            print(
                f"WARNING: final push to https://huggingface.co/{args.hub_repo_id} failed ({e}); "
                f"the local merged checkpoint at {merged_dir} is still intact and eval can proceed "
                f"against it. Common cause: HF private storage quota full -- free up space and "
                f"re-push manually if you need this checkpoint on the Hub.",
                file=sys.stderr,
            )

    print("Done.")


if __name__ == "__main__":
    main()
