"""
Standalone LoRA DPO trainer built on trl's DPOTrainer. Run it once per stage
(1-4), with --data_path set to that stage's build_dpo_stage*.py output and
--model_path set to the previous stage's merged checkpoint.

Adding --reference_free turns the loss into CPO (Contrastive Preference
Optimization) by replacing the reference model with a uniform prior.
DPOConfig/DPOTrainer kwargs change between trl releases, so optional ones are
only passed when the installed version accepts them.

  python train_dpo_standalone.py --model_path <ckpt> \
      --data_path <pairs.json> --output_dir <dir>
"""
import argparse
import inspect
import json

import torch
try:
    # peft does an isinstance(..., DTensor) check that only resolves if this
    # submodule has been imported at least once.
    import torch.distributed.tensor  # noqa: F401
except ImportError:
    pass
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

EOT_TOKEN = "<|eot_id|>"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    """Runs one DPO stage: load the checkpoint, attach LoRA, train on preference pairs, merge.

    DPO learns from pairs, not from single answers. For every prompt it sees a "chosen" and a
    "rejected" answer, and it shifts probability towards the chosen one while a reference copy of
    the model holds it back from drifting too far. beta controls how strong that leash is."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                         help="Base checkpoint, or the previous stage's merged output, e.g. "
                              "<dpo_output_dir>/stage1/merged for Stage 2")
    parser.add_argument("--data_path", required=True,
                         help="A build_dpo_stage*.py output: JSON list of {prompt, chosen, rejected}")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL penalty coefficient")
    parser.add_argument("--rpo_alpha", type=float, default=1.0,
                         help="Weight of an extra NLL loss that keeps the model close to the "
                              "chosen sequence itself, not only to preferring it over the "
                              "rejected one. It reduces unfaithful output that plain DPO does "
                              "not penalize. Set to 0 to turn it off.")
    parser.add_argument("--reference_free", action="store_true",
                         help="Skip the reference-model forward pass and use a uniform prior "
                              "instead, which makes the loss CPO rather than DPO. Uses less "
                              "memory since no second model pass is needed.")
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                         help="Lower than SFT's 1e-5, since DPO training is easier to destabilize")
    parser.add_argument("--num_train_epochs", type=float, default=1.0,
                         help="1 epoch per stage; raise it only if a stage's data is small")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_examples", type=int, default=None,
                         help="Cap dataset size; use a small number for a smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Step 1: load the tokenizer and make sure the turn-end token is the EOS.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if EOT_TOKEN in tokenizer.get_vocab() and tokenizer.eos_token != EOT_TOKEN:
        tokenizer.eos_token = EOT_TOKEN  # EAST-8B ships the token but does not set it as eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: load the checkpoint in bfloat16. Gradient checkpointing recomputes activations in
    # the backward pass instead of storing them, trading speed for memory.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()

    # Step 3: the LoRA settings handed to DPOTrainer below. The base weights stay frozen and only
    # small low-rank matrices next to the attention q/k/v/o and MLP gate/up/down layers are trained.
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    # Step 4: load the preference pairs and put each one into the model's chat format. trl expects
    # the three fields prompt / chosen / rejected.
    with open(args.data_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if args.max_examples:
        pairs = pairs[: args.max_examples]
    print(f"Training on {len(pairs)} preference pairs")

    def to_chat_example(ex):
        """Wrap one pair in the chat template and append the turn-end token to both answers."""
        return {
            "prompt": tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            "chosen": f"{ex['chosen']}{EOT_TOKEN}",
            "rejected": f"{ex['rejected']}{EOT_TOKEN}",
        }

    dataset = Dataset.from_list([to_chat_example(ex) for ex in pairs])

    # Step 5: assemble the DPO config. Because trl's field names move between releases, the
    # optional ones below are only added when the installed version actually accepts them.
    dpo_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        beta=args.beta,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    # Length kwarg names differ between trl versions, so only pass the accepted ones.
    accepted_params = set(inspect.signature(DPOConfig.__init__).parameters)
    for key, value in {"max_length": args.cutoff_len, "max_prompt_length": args.cutoff_len // 2}.items():
        if key in accepted_params:
            dpo_kwargs[key] = value
        else:
            print(f"NOTE: installed trl's DPOConfig has no '{key}' param -- skipping, using its default")

    # Old trl had an rpo_alpha field; newer trl expresses the same thing as
    # loss_type=["sigmoid", "sft"] with loss_weights=[1.0, alpha].
    if args.rpo_alpha and args.rpo_alpha > 0:
        if "rpo_alpha" in accepted_params:
            dpo_kwargs["rpo_alpha"] = args.rpo_alpha
        elif "loss_weights" in accepted_params:
            dpo_kwargs["loss_type"] = ["sigmoid", "sft"]
            dpo_kwargs["loss_weights"] = [1.0, args.rpo_alpha]
        else:
            print(f"WARNING: installed trl's DPOConfig has neither 'rpo_alpha' nor "
                  f"'loss_weights' -- the NLL-anchoring fix is NOT applied this run.")

    if args.reference_free:
        if "reference_free" in accepted_params:
            dpo_kwargs["reference_free"] = True
        else:
            print("WARNING: installed trl's DPOConfig has no 'reference_free' param -- "
                  "falling back to a real reference model this run (still correct, just "
                  "not CPO's claimed 1x memory/FLOPs efficiency).")

    dpo_config = DPOConfig(**dpo_kwargs)

    # DPOTrainer renamed `tokenizer` to `processing_class`, so check which one it takes.
    trainer_kwargs = dict(model=model, args=dpo_config, train_dataset=dataset, peft_config=lora_config)
    trainer_params = set(inspect.signature(DPOTrainer.__init__).parameters)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "Installed trl's DPOTrainer accepts neither 'processing_class' nor 'tokenizer' "
            "-- check `pip show trl` and this script's trl version assumptions"
        )

    # Step 6: the training loop. For each batch DPOTrainer scores the chosen and the rejected
    # answer under both the model and the reference, and updates the LoRA weights so the gap
    # favours the chosen one.
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()

    # Step 7: fold the LoRA weights into the base weights, so the next stage (or eval) can load
    # this directory as an ordinary model.
    merged_dir = f"{args.output_dir.rstrip('/')}/merged"
    print(f"Merging LoRA adapter and saving to {merged_dir}")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print("Done.")


if __name__ == "__main__":
    main()
