#!/bin/bash
# Runs the 4-stage DPO pipeline on top of an SFT checkpoint, then evaluates it:
#   Stage 1: human dialect reference vs. SFT_MODEL's own output
#   Stage 2: human dialect reference vs. the same text with EG<->LB markers swapped
#   Stage 3: human dialect reference vs. the Stage-2 checkpoint's own output
#   Stage 4: human dialect reference vs. perturbed versions of it
# Each stage continues from the previous stage's merged checkpoint.
#
# Run on a Katana GPU node after setup_katana_env.sh and `pip install "trl>=0.15"`.
# Set SFT_MODEL to the checkpoint to start from. Set SMOKE_TEST=1 to try
# Stage 1 on 32 examples first. Set HF_TOKEN and HF_REPO_ID to push each stage.
set -e

cd "$(dirname "$0")/.."

export EAST_CACHE=${EAST_CACHE:-${TMPDIR:-/scratch/$USER}/east_cache}
mkdir -p "$EAST_CACHE/huggingface" "$EAST_CACHE/torch"
export HF_HOME="$EAST_CACHE/huggingface"
export TRANSFORMERS_CACHE="$EAST_CACHE/huggingface"
export HF_HUB_CACHE="$EAST_CACHE/huggingface/hub"
export TORCH_HOME="$EAST_CACHE/torch"

SFT_MODEL=${SFT_MODEL:?"Set SFT_MODEL to the merged Phase-2 checkpoint dir, e.g. \$EAST_CACHE/east-arabic-eg-lora_replay/merged"}
DATA_PATH=${DATA_PATH:-data/mt_data/train_data/Arabic-EG-SiMT-OMT.json}
DIALECT=${DIALECT:-EG}
DPO_ROOT=${DPO_ROOT:-$EAST_CACHE/east-arabic-dpo}
DPO_DATA_ROOT=${DPO_DATA_ROOT:-data/mt_data/dpo_data}
COMET_CKPT=${COMET_CKPT:-$(python -c "from comet import download_model; print(download_model('Unbabel/wmt22-comet-da'))")}

mkdir -p "${DPO_DATA_ROOT}"

# Pushes each stage's merged checkpoint to its own repo, "${HF_REPO_ID}-stageN".
# Does nothing unless both HF_TOKEN and HF_REPO_ID are set. Worth enabling, since
# $TMPDIR is wiped when the job ends.
push_to_hub() {
    local merged_dir="$1" stage_label="$2"
    if [ -z "${HF_TOKEN:-}" ] || [ -z "${HF_REPO_ID:-}" ]; then
        return 0
    fi
    local repo_id="${HF_REPO_ID}-${stage_label}"
    echo "=== Pushing ${stage_label} checkpoint to https://huggingface.co/${repo_id} ==="
    python - "${merged_dir}" "${repo_id}" <<'PY'
import sys
from huggingface_hub import HfApi
merged_model, repo_id = sys.argv[1], sys.argv[2]
api = HfApi()
repo_id = api.create_repo(repo_id=repo_id, private=True, exist_ok=True).repo_id
api.upload_folder(repo_id=repo_id, folder_path=merged_model)
print(f"Uploaded to {repo_id}")
PY
}

if [ "${SMOKE_TEST:-0}" = "1" ]; then
    echo "=== SMOKE TEST: Stage 1 only, 32 examples, 1 quick pass ==="
    python ./east_scripts/data/generate_candidates.py \
        --model_path "${SFT_MODEL}" --data_path "${DATA_PATH}" \
        --output "${DPO_DATA_ROOT}/smoke_candidates.json" --max_examples 32
    python ./east_scripts/data/build_dpo_stage1_msa_vs_dialect.py \
        --candidates_path "${DPO_DATA_ROOT}/smoke_candidates.json" \
        --output "${DPO_DATA_ROOT}/smoke_pairs.json"
    python ./east_scripts/train/train_dpo_standalone.py \
        --model_path "${SFT_MODEL}" --data_path "${DPO_DATA_ROOT}/smoke_pairs.json" \
        --output_dir "${DPO_ROOT}-smoke" \
        --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --logging_steps 1
    echo "Smoke test passed. Re-run without SMOKE_TEST=1 for the full pipeline."
    exit 0
fi

echo "=== Stage 1: MSA vs Dialect ==="
echo "--- generating SFT checkpoint's own current candidates (on-policy negatives) ---"
python ./east_scripts/data/generate_candidates.py \
    --model_path "${SFT_MODEL}" \
    --data_path "${DATA_PATH}" \
    --output "${DPO_DATA_ROOT}/stage1_candidates.json"

python ./east_scripts/data/build_dpo_stage1_msa_vs_dialect.py \
    --candidates_path "${DPO_DATA_ROOT}/stage1_candidates.json" \
    --output "${DPO_DATA_ROOT}/stage1_pairs.json"

python ./east_scripts/train/train_dpo_standalone.py \
    --model_path "${SFT_MODEL}" \
    --data_path "${DPO_DATA_ROOT}/stage1_pairs.json" \
    --output_dir "${DPO_ROOT}/stage1"

push_to_hub "${DPO_ROOT}/stage1/merged" "stage1"

echo "=== Stage 2: Dialect A vs Dialect B ==="
python ./east_scripts/data/build_dpo_stage2_dialect_vs_dialect.py \
    --data_path "${DATA_PATH}" \
    --src_dialect "${DIALECT}" \
    --dst_dialect LB \
    --output "${DPO_DATA_ROOT}/stage2_pairs.json"

python ./east_scripts/train/train_dpo_standalone.py \
    --model_path "${DPO_ROOT}/stage1/merged" \
    --data_path "${DPO_DATA_ROOT}/stage2_pairs.json" \
    --output_dir "${DPO_ROOT}/stage2"

push_to_hub "${DPO_ROOT}/stage2/merged" "stage2"

echo "=== Stage 3: Human vs Current-Model Output ==="
python ./east_scripts/data/generate_candidates.py \
    --model_path "${DPO_ROOT}/stage2/merged" \
    --data_path "${DATA_PATH}" \
    --output "${DPO_DATA_ROOT}/stage3_candidates.json"

python ./east_scripts/data/build_dpo_stage3_human_vs_model.py \
    --candidates_path "${DPO_DATA_ROOT}/stage3_candidates.json" \
    --output "${DPO_DATA_ROOT}/stage3_pairs.json"

python ./east_scripts/train/train_dpo_standalone.py \
    --model_path "${DPO_ROOT}/stage2/merged" \
    --data_path "${DPO_DATA_ROOT}/stage3_pairs.json" \
    --output_dir "${DPO_ROOT}/stage3"

push_to_hub "${DPO_ROOT}/stage3/merged" "stage3"

echo "=== Stage 4: Perturbation Hardening ==="
python ./east_scripts/data/build_dpo_stage4_perturbation.py \
    --data_path "${DATA_PATH}" \
    --dialect "${DIALECT}" \
    --output "${DPO_DATA_ROOT}/stage4_pairs.json"

python ./east_scripts/train/train_dpo_standalone.py \
    --model_path "${DPO_ROOT}/stage3/merged" \
    --data_path "${DPO_DATA_ROOT}/stage4_pairs.json" \
    --output_dir "${DPO_ROOT}/stage4"

push_to_hub "${DPO_ROOT}/stage4/merged" "stage4-final"

FINAL_MODEL="${DPO_ROOT}/stage4/merged"

echo "=== Eval: held-out EG/Commerce test set, all latencies ==="
for latency in "low" "low-medium" "medium" "medium-high" "high"; do
    RESULT_PATH=results/alexandria_eg_commerce_dpo/${latency}
    echo "--- latency: ${latency} -> ${RESULT_PATH} ---"
    python ./east_scripts/eval/simuleval_standalone.py \
        --model_path "${FINAL_MODEL}" \
        --data_path data/mt_data/test_data/alexandria.test.eg.commerce_and_transactions.en2ar.json \
        --output_dir "${RESULT_PATH}" \
        --max_new_tokens 256 \
        --latency "${latency}" \
        --num_beams 5 \
        --comet_ckpt_path "${COMET_CKPT}"
done

echo "=== Eval: cross-dialect check on Lebanese held-out test set ==="
python ./east_scripts/eval/simuleval_standalone.py \
    --model_path "${FINAL_MODEL}" \
    --data_path data/mt_data/test_data/alexandria.test.lb.commerce_and_transactions.en2ar.json \
    --output_dir results/alexandria_lb_commerce_dpo/medium \
    --max_new_tokens 256 \
    --latency medium \
    --num_beams 5 \
    --comet_ckpt_path "${COMET_CKPT}"

echo "=== Eval: WMT22 De-En regression check (catastrophic forgetting) ==="
python ./east_scripts/eval/simuleval_standalone.py \
    --model_path "${FINAL_MODEL}" \
    --data_path data/mt_data/test_data/wmt22.test.de-en.json \
    --output_dir results/wmt22_de-en_dpo_regression_check/medium \
    --max_new_tokens 256 \
    --latency medium \
    --num_beams 5 \
    --max_examples 100 \
    --comet_ckpt_path "${COMET_CKPT}"

echo "=== MSA-contamination check (proxy metric, EG only) ==="
python ./east_scripts/data/measure_dialect_contamination.py \
    --predictions_path results/alexandria_eg_commerce_dpo/medium/prediction.json \
    --dialect "${DIALECT}"

echo "Done. Compare against Phase-2 SFT with:"
echo "  python east_scripts/eval/summarize_alexandria_results.py --result_root sft=results/alexandria_eg_commerce_finetuned dpo=results/alexandria_eg_commerce_dpo"
