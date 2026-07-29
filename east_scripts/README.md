# Egyptian Arabic simultaneous MT

Adapting the EAST simultaneous-translation framework to **English → spoken Egyptian Arabic** (the
everyday dialect, not the formal MSA taught in school). We test on the **Alexandria** dataset (English
sentences with human Egyptian translations). Main findings: [RESULTS.md](RESULTS.md).

Training and evaluation run on the **Katana** HPC cluster (PBS Pro). Datasets and model weights are not
in this repo: data goes under `data/mt_data/` locally, models are on Hugging Face.

## Folders

| Folder | What is in it |
|---|---|
| `data/` | Build the training and test data: Alexandria → EAST interleaved SiMT + offline format, semantic chunking, DPO pair building, replay mix |
| `train/` | Fine-tuning: Stage I (full), Stage II (LoRA), and DPO |
| `eval/` | Simultaneous inference, quality scores (BLEU, spBLEU, chrF++, COMET, BERTScore), latency (AL), and analysis |
| `jobs/` | Katana job scripts (`.pbs`) and the drivers that chain several steps together |

`env.sh` and `setup_katana_env.sh` build the environment inside each job; `requirements_eval.txt` pins
the evaluation packages.

## Main pipeline

1. **Data** - `data/prepare_alexandria_test_data.py` (held-out test set),
   `data/build_arabic_simt_sft_data.py` (training data; add `--chunks_path` to use LLM chunks from
   `data/generate_semantic_chunks.py`).
2. **Train** - `train/finetune_nilechat_stage1_general.py` (Stage I, teaches the read/write tokens),
   then `train/finetune_lora_stage2.py` (Stage II LoRA, adapts to the dialect).
3. **Preference tuning (optional)** - `data/build_dpo_stage{1..4}_*.py` build the preference pairs,
   `train/train_dpo_standalone.py` trains on them. This is what cut the formal-Arabic leakage.
4. **Evaluate** - `eval/simuleval_standalone.py` at 5 latency settings, then
   `eval/summarize_alexandria_results.py` for the results table.

`jobs/run_full_cascade_comparison.sh` runs steps 1-4 end to end for one configuration.

## Running a job

Submit from the repo root, with the walltime and GPU on the command line:

```bash
qsub -l walltime=6:00:00 -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_model=H200 \
     -v <VARS> east_scripts/jobs/submit_full_cascade.pbs
```

The Hugging Face token is read from `~/.hf_token` at runtime, never hardcoded.
