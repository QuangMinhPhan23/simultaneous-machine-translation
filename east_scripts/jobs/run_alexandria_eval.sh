#!/bin/bash
# Evaluates a checkpoint on the Alexandria EG English->Egyptian Arabic test set
# at all five latency settings, reporting BLEU/COMET/AL/LAAL.
# Run on a Katana GPU node after setup_katana_env.sh.
# Override MODEL_PATH, DATASET, BEAM or RESULT_ROOT to change what is evaluated.
# BLEURT is off by default; set BLEURT_CKPT to a BLEURT-20 directory to include it.
set -e

cd "$(dirname "$0")/.."

export EAST_CACHE=${EAST_CACHE:-${TMPDIR:-/scratch/$USER}/east_cache}
mkdir -p "$EAST_CACHE/huggingface" "$EAST_CACHE/torch"
export HF_HOME="$EAST_CACHE/huggingface"
export TRANSFORMERS_CACHE="$EAST_CACHE/huggingface"
export HF_HUB_CACHE="$EAST_CACHE/huggingface/hub"
export TORCH_HOME="$EAST_CACHE/torch"

MODEL_PATH=${MODEL_PATH:-biaofu-xmu/EAST-8B}   # downloads from the HF Hub into $HF_HOME
DATASET=${DATASET:-data/mt_data/test_data/alexandria.test.eg.commerce_and_transactions.en2ar.json}
BEAM=${BEAM:-5}
BLEURT_CKPT=${BLEURT_CKPT:-}   # empty by default; set to a BLEURT-20 dir to enable
COMET_CKPT=${COMET_CKPT:-$(python -c "from comet import download_model; print(download_model('Unbabel/wmt22-comet-da'))")}
RESULT_ROOT=${RESULT_ROOT:-results/alexandria_eg_commerce}

for latency in "low" "low-medium" "medium" "medium-high" "high"; do
    RESULT_PATH=${RESULT_ROOT}/${latency}

    echo "=== latency: ${latency} -> ${RESULT_PATH} ==="
    python ./east_scripts/eval/simuleval_standalone.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATASET}" \
        --output_dir "${RESULT_PATH}" \
        --max_new_tokens 256 \
        --latency "${latency}" \
        --num_beams "${BEAM}" \
        ${BLEURT_CKPT:+--bleurt_ckpt_path "${BLEURT_CKPT}"} \
        --comet_ckpt_path "${COMET_CKPT}"
done
