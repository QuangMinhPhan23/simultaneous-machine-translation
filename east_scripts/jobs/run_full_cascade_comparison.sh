#!/bin/bash
# Runs the chunk-vs-turn granularity comparison: Nile-Chat Stage I (once), then
# for each model x data variant, build the training data, run LoRA Stage II,
# eval at 5 latencies, and run a WMT22 regression check for EAST-8B.
# Run on a Katana GPU node after setup_katana_env.sh.
#
# Set HF_TOKEN (and optionally HF_REPO_PREFIX) to push every checkpoint to the
# HF Hub. Set SMOKE_TEST=1 to run the whole pipeline on tiny slices first.
#
# Variables for splitting the work across PBS jobs (see submit_full_cascade.pbs):
#   RUN_DATA_BUILD=0   skip Step 0 (data building / chunk generation)
#   RUN_STAGE1=0       skip Step 1; then set NILECHAT_STAGE1_MODEL_OVERRIDE to a
#                      path or HF repo id so Step 2's Nile-Chat leg has a checkpoint
#   RUN_STAGE2=0       skip Step 2
#   ONLY_MODEL=east8b|nilechat|gemma   run Step 2 for one model only
#   VARIANTS="..."     run a subset of the data variants
set -e

cd "$(dirname "$0")/.."

export EAST_CACHE=${EAST_CACHE:-${TMPDIR:-/scratch/$USER}/east_cache}
mkdir -p "$EAST_CACHE/huggingface" "$EAST_CACHE/torch"
export HF_HOME="$EAST_CACHE/huggingface"
export TRANSFORMERS_CACHE="$EAST_CACHE/huggingface"
export HF_HUB_CACHE="$EAST_CACHE/huggingface/hub"
export TORCH_HOME="$EAST_CACHE/torch"

COMET_CKPT=${COMET_CKPT:-$(python -c "from comet import download_model; print(download_model('Unbabel/wmt22-comet-da'))")}
HF_REPO_PREFIX=${HF_REPO_PREFIX:-Henry236}
CHUNKERS=${CHUNKERS:-"llama gemma nilechat"}
VARIANTS=${VARIANTS:-"heuristic turn chunk-llama chunk-gemma chunk-nilechat"}
SMOKE_TEST=${SMOKE_TEST:-0}
RUN_DATA_BUILD=${RUN_DATA_BUILD:-1}
RUN_STAGE1=${RUN_STAGE1:-1}
RUN_STAGE2=${RUN_STAGE2:-1}
ONLY_MODEL=${ONLY_MODEL:-}
NILECHAT_STAGE1_MODEL_OVERRIDE=${NILECHAT_STAGE1_MODEL_OVERRIDE:-}

if [ "$SMOKE_TEST" = "1" ]; then
    echo ">>> SMOKE_TEST=1: capping every stage to --max_examples 32 (5 at eval time), 1 epoch, 1 latency; HF auto-push disabled <<<"
    STAGE1_EXTRA_ARGS="--max_examples 32 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --logging_steps 1"
    STAGE2_EXTRA_ARGS="--max_examples 32 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --logging_steps 1"
    CHUNK_GEN_EXTRA_ARGS="--max_examples 8"
    EVAL_LATENCIES="medium"
    # Eval is slow per example, so cap it too or the smoke test is not quick.
    EVAL_MAX_EXAMPLES_ARGS="--max_examples 5"
    SMOKE_SUFFIX="-smoke"
    # HF_TOKEN stays set because reading gated base models needs it; only HF_REPO_ID is unset below.
else
    STAGE1_EXTRA_ARGS=""
    STAGE2_EXTRA_ARGS=""
    CHUNK_GEN_EXTRA_ARGS=""
    EVAL_LATENCIES="low low-medium medium medium-high high"
    EVAL_MAX_EXAMPLES_ARGS=""
    SMOKE_SUFFIX=""
fi

data_path_for_variant() {
    case "$1" in
        heuristic) echo "data/mt_data/train_data/Arabic-EG-SiMT-OMT${SMOKE_SUFFIX}.json" ;;
        turn) echo "data/mt_data/train_data/Arabic-EG-SiMT-OMT-turn${SMOKE_SUFFIX}.json" ;;
        chunk-llama) echo "data/mt_data/train_data/Arabic-EG-SiMT-OMT-chunk-llama${SMOKE_SUFFIX}.json" ;;
        chunk-gemma) echo "data/mt_data/train_data/Arabic-EG-SiMT-OMT-chunk-gemma${SMOKE_SUFFIX}.json" ;;
        chunk-nilechat) echo "data/mt_data/train_data/Arabic-EG-SiMT-OMT-chunk-nilechat${SMOKE_SUFFIX}.json" ;;
        *) echo "unknown variant: $1" >&2; exit 1 ;;
    esac
}

if [ "$RUN_DATA_BUILD" = "1" ]; then
    echo "=== Step 0: build any missing training-data variants ==="
    # SMOKE_SUFFIX keeps tiny smoke-run files off the real filenames, so they
    # can never satisfy the "already exists" check of a later real run.
    HEURISTIC_DATA_FILE="data/mt_data/train_data/Arabic-EG-SiMT-OMT${SMOKE_SUFFIX}.json"
    TURN_DATA_FILE="data/mt_data/train_data/Arabic-EG-SiMT-OMT-turn${SMOKE_SUFFIX}.json"
    [ -f "$HEURISTIC_DATA_FILE" ] || \
        python ./east_scripts/data/build_arabic_simt_sft_data.py \
            --output "${HEURISTIC_DATA_FILE}"
    [ -f "$TURN_DATA_FILE" ] || \
        python ./east_scripts/data/build_arabic_simt_sft_data.py --granularity turn \
            --output "${TURN_DATA_FILE}"
    for chunker in $CHUNKERS; do
        CHUNKS_FILE="data/mt_data/train_data/chunks-${chunker}${SMOKE_SUFFIX}.json"
        DATA_FILE="data/mt_data/train_data/Arabic-EG-SiMT-OMT-chunk-${chunker}${SMOKE_SUFFIX}.json"
        # Not exists-gated on purpose: --resume only retries failed turns, so
        # rerunning is cheap and DATA_FILE never goes stale against CHUNKS_FILE.
        echo "--- generating/retrying semantic chunks with ${chunker} ---"
        python ./east_scripts/data/generate_semantic_chunks.py --chunker_model "${chunker}" --output "${CHUNKS_FILE}" --resume ${CHUNK_GEN_EXTRA_ARGS}
        python ./east_scripts/data/build_arabic_simt_sft_data.py --chunks_path "${CHUNKS_FILE}" --output "${DATA_FILE}"
    done
else
    echo "=== Step 0: skipped (RUN_DATA_BUILD=0) ==="
fi

# Control leg: same cascade as nilechat, but Stage I starts from raw
# google/gemma-3-4b-it, which has no Egyptian pre-training. Runs only when
# ONLY_MODEL=gemma. Comparing gemma vs nilechat results estimates what the
# extra Egyptian pre-training is worth.
GEMMA_STAGE1_MODEL="${GEMMA_STAGE1_MODEL_OVERRIDE:-}"
if [ "$ONLY_MODEL" = "gemma" ]; then
    if [ "$RUN_STAGE1" = "1" ]; then
        echo "=== Step 1 (gemma control): Stage I on raw google/gemma-3-4b-it ==="
        GEMMA_STAGE1_DIR="${EAST_CACHE}/gemma-stage1-general${SMOKE_SUFFIX}"
        if [ -n "${HF_TOKEN:-}" ] && [ "$SMOKE_TEST" != "1" ]; then
            export HF_REPO_ID="${HF_REPO_PREFIX}/gemma-eg-stage1-general"
        else
            unset HF_REPO_ID
        fi
        python ./east_scripts/train/finetune_nilechat_stage1_general.py \
            --model_path google/gemma-3-4b-it \
            --output_dir "${GEMMA_STAGE1_DIR}" ${STAGE1_EXTRA_ARGS}
        GEMMA_STAGE1_MODEL="${GEMMA_STAGE1_DIR}/final"
    else
        : "${GEMMA_STAGE1_MODEL_OVERRIDE:?RUN_STAGE1=0 with ONLY_MODEL=gemma needs GEMMA_STAGE1_MODEL_OVERRIDE (a prior gemma Stage I checkpoint)}"
    fi
    NILECHAT_STAGE1_MODEL=""   # not used in the gemma control leg
elif [ "$RUN_STAGE1" = "1" ]; then
    echo "=== Step 1: Nile-Chat Stage I (general SiMT+OMT mechanism, once) ==="
    NILECHAT_STAGE1_DIR="${EAST_CACHE}/nilechat-stage1-general${SMOKE_SUFFIX}"
    if [ -n "${HF_TOKEN:-}" ] && [ "$SMOKE_TEST" != "1" ]; then
        export HF_REPO_ID="${HF_REPO_PREFIX}/nilechat-eg-stage1-general"
    else
        unset HF_REPO_ID
    fi
    python ./east_scripts/train/finetune_nilechat_stage1_general.py \
        --output_dir "${NILECHAT_STAGE1_DIR}" ${STAGE1_EXTRA_ARGS}
    NILECHAT_STAGE1_MODEL="${NILECHAT_STAGE1_DIR}/final"
else
    if [ "$RUN_STAGE2" = "1" ] && [ "$ONLY_MODEL" != "east8b" ]; then
        # Only needed when Step 2 runs a Nile-Chat leg.
        : "${NILECHAT_STAGE1_MODEL_OVERRIDE:?RUN_STAGE1=0 requires NILECHAT_STAGE1_MODEL_OVERRIDE (local path or HF Hub repo id from a prior Stage I job) so the Nile-Chat Step 2 leg has a checkpoint to continue from}"
    fi
    echo "=== Step 1: skipped (RUN_STAGE1=0) -- using NILECHAT_STAGE1_MODEL_OVERRIDE=${NILECHAT_STAGE1_MODEL_OVERRIDE:-<unset, not needed for ONLY_MODEL=east8b>} ==="
    NILECHAT_STAGE1_MODEL="$NILECHAT_STAGE1_MODEL_OVERRIDE"
fi

run_stage2_and_eval() {
    local model_label=$1 model_path=$2 eot_token=$3 data_path=$4 variant=$5 with_replay=$6

    FT_OUTPUT_DIR="${EAST_CACHE}/${model_label}-eg-lora-${variant}${SMOKE_SUFFIX}"
    if [ -n "${HF_TOKEN:-}" ] && [ "$SMOKE_TEST" != "1" ]; then
        export HF_REPO_ID="${HF_REPO_PREFIX}/${model_label}-eg-lora-${variant}"
    else
        unset HF_REPO_ID
    fi

    echo "--- Stage II: ${model_label} x ${variant} ---"
    python ./east_scripts/train/finetune_lora_stage2.py \
        --model_path "${model_path}" \
        --eot_token "${eot_token}" \
        --data_path "${data_path}" \
        --output_dir "${FT_OUTPUT_DIR}" ${STAGE2_EXTRA_ARGS}

    MERGED_MODEL="${FT_OUTPUT_DIR}/merged"
    for latency in $EVAL_LATENCIES; do
        RESULT_PATH="results/granularity_comparison${SMOKE_SUFFIX}/${model_label}/${variant}/${latency}"
        python ./east_scripts/eval/simuleval_standalone.py \
            --model_path "${MERGED_MODEL}" \
            --data_path data/mt_data/test_data/alexandria.test.eg.commerce_and_transactions.en2ar.json \
            --output_dir "${RESULT_PATH}" \
            --max_new_tokens 256 \
            --latency "${latency}" \
            --num_beams 5 \
            --eot_token "${eot_token}" \
            --comet_ckpt_path "${COMET_CKPT}" ${EVAL_MAX_EXAMPLES_ARGS}
    done

    if [ "$with_replay" = "yes" ]; then
        WMT22_MAX_EXAMPLES_ARGS=${EVAL_MAX_EXAMPLES_ARGS:---max_examples 100}
        python ./east_scripts/eval/simuleval_standalone.py \
            --model_path "${MERGED_MODEL}" \
            --data_path data/mt_data/test_data/wmt22.test.de-en.json \
            --output_dir "results/wmt22_de-en_${model_label}_${variant}_regression${SMOKE_SUFFIX}/medium" \
            --max_new_tokens 256 --latency medium --num_beams 5 \
            --eot_token "${eot_token}" \
            --comet_ckpt_path "${COMET_CKPT}" ${WMT22_MAX_EXAMPLES_ARGS}
    fi
}

if [ "$RUN_STAGE2" = "1" ]; then
    echo "=== Step 2: Stage II sweep -- variants: [${VARIANTS}], model(s): [${ONLY_MODEL:-east8b nilechat}] ==="
    for variant in $VARIANTS; do
        DATA_FILE=$(data_path_for_variant "$variant")

        if [ -z "$ONLY_MODEL" ] || [ "$ONLY_MODEL" = "east8b" ]; then
            # EAST-8B trains on the replay-mixed data, which limits forgetting of its old languages.
            REPLAY_FILE="${DATA_FILE%.json}-with-replay.json"
            case "$variant" in
                chunk-*)
                    # Always rebuilt, because chunk-* DATA_FILEs can change between runs.
                    python ./east_scripts/data/build_replay_mix.py --arabic_path "${DATA_FILE}" --output "${REPLAY_FILE}"
                    ;;
                *)
                    [ -f "$REPLAY_FILE" ] || python ./east_scripts/data/build_replay_mix.py --arabic_path "${DATA_FILE}" --output "${REPLAY_FILE}"
                    ;;
            esac
            run_stage2_and_eval "east8b" "biaofu-xmu/EAST-8B" "<|eot_id|>" "${REPLAY_FILE}" "${variant}" "yes"
        fi

        if [ -z "$ONLY_MODEL" ] || [ "$ONLY_MODEL" = "nilechat" ]; then
            # No replay here: this Stage I checkpoint has no earlier languages to forget.
            run_stage2_and_eval "nilechat" "${NILECHAT_STAGE1_MODEL}" "<end_of_turn>" "${DATA_FILE}" "${variant}" "no"
        fi

        if [ "$ONLY_MODEL" = "gemma" ]; then
            run_stage2_and_eval "gemma" "${GEMMA_STAGE1_MODEL}" "<end_of_turn>" "${DATA_FILE}" "${variant}" "no"
        fi
    done
else
    echo "=== Step 2: skipped (RUN_STAGE2=0) ==="
fi

echo "Done. Compare with, e.g.:"
echo "  python east_scripts/eval/summarize_alexandria_results.py --result_root \\"
echo "    east8b_heuristic=results/granularity_comparison/east8b/heuristic \\"
echo "    east8b_turn=results/granularity_comparison/east8b/turn \\"
echo "    east8b_chunk-llama=results/granularity_comparison/east8b/chunk-llama \\"
echo "    east8b_chunk-gemma=results/granularity_comparison/east8b/chunk-gemma \\"
echo "    east8b_chunk-nilechat=results/granularity_comparison/east8b/chunk-nilechat \\"
echo "    nilechat_heuristic=results/granularity_comparison/nilechat/heuristic \\"
echo "    nilechat_turn=results/granularity_comparison/nilechat/turn \\"
echo "    nilechat_chunk-llama=results/granularity_comparison/nilechat/chunk-llama \\"
echo "    nilechat_chunk-gemma=results/granularity_comparison/nilechat/chunk-gemma \\"
echo "    nilechat_chunk-nilechat=results/granularity_comparison/nilechat/chunk-nilechat"
