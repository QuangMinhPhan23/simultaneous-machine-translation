"""
Stage I: full-parameter SFT that teaches a base chat model the adaptive
read/write mechanism (<|end-of-read|>/<|end-of-write|>).

Trains only on general-language SiMT+OMT data (SiMT-Multi-90K +
Off-Multi-120K); Arabic comes later in Stage II. --n_simt/--n_omt cap how many
rows are used, so a run fits inside the job walltime. The job checkpoints every
--save_steps and resumes from --output_dir automatically if it is killed.

  python finetune_nilechat_stage1_general.py --output_dir <dir>
"""
import argparse
import json
import os
import random
import sys

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from build_replay_mix import build_omt_replay, build_simt_replay

EOT_TOKEN = "<end_of_turn>"
NEW_SPECIAL_TOKENS = ["<|end-of-read|>", "<|end-of-write|>"]


class AlpacaSFTDataset(Dataset):
    def __init__(self, examples, tokenizer, cutoff_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len

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
        full_text = f"{prompt_text}{ex['output']}{EOT_TOKEN}"

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


def build_general_stage1_data(simt_multi_path_hint, omt_path, n_simt, n_omt, seed):
    # build_simt_replay() pulls SiMT-Multi-90K from the HF Hub, so the path hint is unused.
    del simt_multi_path_hint
    simt_examples = build_simt_replay(n_simt, seed)
    omt_examples = build_omt_replay(omt_path, n_omt, seed)
    combined = simt_examples + omt_examples
    random.Random(seed).shuffle(combined)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="MBZUAI-Paris/Nile-Chat-4B")
    parser.add_argument("--omt_path", default="data/mt_data/train_data/Off-Multi-120K.json")
    parser.add_argument("--n_simt", type=int, default=12_000,
                         help="Cap on SiMT-Multi-90K rows; raise it if you have a longer walltime")
    parser.add_argument("--n_omt", type=int, default=6_000,
                         help="Cap on Off-Multi-120K rows; raise it if you have a longer walltime")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50,
                         help="Checkpoint every N steps so a killed job can resume")
    parser.add_argument("--max_examples", type=int, default=None,
                         help="Cap total dataset size after combining; use a small number for a smoke test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub_repo_id", default=os.environ.get("HF_REPO_ID"),
                         help="If set (or $HF_REPO_ID), push checkpoints to this HF Hub repo. "
                              "Requires $HF_TOKEN to be exported.")
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    num_added = tokenizer.add_special_tokens({
        "eos_token": EOT_TOKEN,
        "additional_special_tokens": NEW_SPECIAL_TOKENS,
    })
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Added {num_added} new tokens to the vocabulary (expect {len(NEW_SPECIAL_TOKENS)})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    model.gradient_checkpointing_enable()

    examples = build_general_stage1_data(None, args.omt_path, args.n_simt, args.n_omt, args.seed)
    random.shuffle(examples)
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(f"Training on {len(examples)} general (non-Arabic) SiMT+OMT examples")

    dataset = AlpacaSFTDataset(examples, tokenizer, args.cutoff_len)

    push_to_hub = bool(args.hub_repo_id)
    if push_to_hub and not os.environ.get("HF_TOKEN"):
        print("WARNING: --hub_repo_id/HF_REPO_ID set but $HF_TOKEN is not -- disabling auto-push.", file=sys.stderr)
        push_to_hub = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        push_to_hub=push_to_hub,
        hub_model_id=args.hub_repo_id if push_to_hub else None,
        # "checkpoint" keeps only the latest one on the Hub instead of every one, to save quota.
        hub_strategy="checkpoint" if push_to_hub else "every_save",
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

    final_dir = f"{args.output_dir.rstrip('/')}/final"
    print(f"Saving Stage I checkpoint to {final_dir}")
    # Some transformers versions push inside save_model(); turn it off so a
    # failed push cannot break this local save.
    trainer.args.push_to_hub = False
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.args.push_to_hub = push_to_hub

    if push_to_hub:
        # Not trainer.push_to_hub(): that would keep the "final/" subfolder as a
        # remote prefix, and from_pretrained(repo_id) only reads the repo root.
        print(f"Pushing final Stage I checkpoint to https://huggingface.co/{args.hub_repo_id}")
        try:
            from huggingface_hub import HfApi
            HfApi().upload_folder(
                repo_id=args.hub_repo_id, folder_path=final_dir,
                commit_message="Stage I: general SiMT+OMT mechanism training complete",
            )
        except Exception as e:
            print(
                f"WARNING: final push to https://huggingface.co/{args.hub_repo_id} failed ({e}); "
                f"the local checkpoint at {final_dir} is still intact. Common cause: HF private "
                f"storage quota full -- free up space (see experiment_dpo.md's repro list) and "
                f"re-push manually, or resubmit this job once space is freed.",
                file=sys.stderr,
            )

    print("Done.")


if __name__ == "__main__":
    main()
