"""
Mix the Egyptian Arabic SiMT+OMT training data with a replay sample of the paper's original
training data (SiMT-Multi-90K + Off-Multi-120K), so fine-tuning on Arabic alone does not make
the model forget the languages it already knew.

SiMT-Multi-90K ships pre-chunked rows instead of the alpaca schema, so those rows go through
interleave_chunks() first. Off-Multi-120K is already alpaca and is sampled directly.
"""
import argparse
import json
import random

from datasets import load_dataset

from build_arabic_simt_sft_data import SIMT_INSTRUCTION, interleave_chunks


def build_simt_replay(n_samples, seed):
    ds = load_dataset("biaofu-xmu/SiMT-Multi-90K")["train"]
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_samples, len(ds)))

    examples = []
    for idx in indices:
        row = ds[idx]
        output = interleave_chunks(row["source_chunks"], row["target_chunks"])
        if not output:
            continue
        examples.append(
            {
                "instruction": SIMT_INSTRUCTION.format(
                    src_lang=row["src_lang"], tgt_lang=row["tgt_lang"], latency=row["latency"]
                ),
                "input": "",
                "output": output,
                "src_lang": row["src_lang"],
                "tgt_lang": row["tgt_lang"],
                "latency": row["latency"],
            }
        )
    return examples


def build_omt_replay(omt_path, n_samples, seed):
    with open(omt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed + 1)
    return rng.sample(data, min(n_samples, len(data)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arabic_path", default="data/mt_data/train_data/Arabic-EG-SiMT-OMT.json")
    parser.add_argument("--omt_replay_path", default="data/mt_data/train_data/Off-Multi-120K.json")
    parser.add_argument("--n_simt_replay", type=int, default=4000)
    parser.add_argument("--n_omt_replay", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/mt_data/train_data/Arabic-EG-SiMT-OMT-with-replay.json")
    args = parser.parse_args()

    with open(args.arabic_path, "r", encoding="utf-8") as f:
        arabic_examples = json.load(f)

    simt_replay = build_simt_replay(args.n_simt_replay, args.seed)
    omt_replay = build_omt_replay(args.omt_replay_path, args.n_omt_replay, args.seed)

    combined = arabic_examples + simt_replay + omt_replay
    random.Random(args.seed).shuffle(combined)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(combined)} examples to {args.output} "
        f"({len(arabic_examples)} Arabic + {len(simt_replay)} SiMT replay + {len(omt_replay)} OMT replay)"
    )
