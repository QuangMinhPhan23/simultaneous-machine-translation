#!/bin/bash
# Submits the whole granularity cascade as dependent PBS jobs, so you do not
# have to type 14 qsub commands and track job ids by hand:
#   - 3 chunk-generation jobs and 1 Nile-Chat Stage I job, all independent
#   - 10 Stage II jobs, each waiting on the jobs that produce what it needs
#
# Run from a Katana login node; it only calls qsub, so it needs no GPU.
# Each submitted job reads HF_TOKEN from ~/.hf_token, so create that first.
#
# Usage: bash east_scripts/jobs/submit_full_cascade_jobs.sh
# For a subset, set VARIANTS or HF_REPO_PREFIX, e.g.:
#   VARIANTS="heuristic turn" bash east_scripts/jobs/submit_full_cascade_jobs.sh
set -e

cd "$(dirname "$0")/.."

HF_REPO_PREFIX=${HF_REPO_PREFIX:-Henry236}
VARIANTS=${VARIANTS:-"heuristic turn chunk-llama chunk-gemma chunk-nilechat"}

qsub_id() {
    # qsub prints "<id>.<host>"; keep the whole string for -W depend=afterok:<id>.
    "$@" | tr -d '\n'
}

echo "=== Submitting chunk-generation jobs (llama, gemma, nilechat -- independent) ==="
declare -A CHUNK_JOB
for chunker in llama gemma nilechat; do
    CHUNK_JOB[$chunker]=$(qsub_id qsub -v PHASE=chunkgen,CHUNKER=${chunker},HF_REPO_PREFIX=${HF_REPO_PREFIX} east_scripts/jobs/submit_full_cascade.pbs)
    echo "  ${chunker}: ${CHUNK_JOB[$chunker]}"
done

echo "=== Submitting Nile-Chat Stage I job (independent) ==="
STAGE1_JOB=$(qsub_id qsub -v PHASE=stage1,HF_REPO_PREFIX=${HF_REPO_PREFIX} east_scripts/jobs/submit_full_cascade.pbs)
echo "  stage1: ${STAGE1_JOB}"
NILECHAT_STAGE1_REPO="${HF_REPO_PREFIX}/nilechat-eg-stage1-general"

# Every chunk-generation job also builds the heuristic/turn data, so those two
# variants just borrow the llama job as their data dependency.
data_dep_for_variant() {
    case "$1" in
        heuristic|turn) echo "${CHUNK_JOB[llama]}" ;;
        chunk-llama) echo "${CHUNK_JOB[llama]}" ;;
        chunk-gemma) echo "${CHUNK_JOB[gemma]}" ;;
        chunk-nilechat) echo "${CHUNK_JOB[nilechat]}" ;;
        *) echo "unknown variant: $1" >&2; exit 1 ;;
    esac
}

echo "=== Submitting Stage II jobs (one per model x variant, ${VARIANTS}) ==="
for variant in $VARIANTS; do
    data_dep=$(data_dep_for_variant "$variant")

    east8b_job=$(qsub_id qsub -W depend=afterok:${data_dep} \
        -v PHASE=stage2,MODEL_LABEL=east8b,VARIANT=${variant},HF_REPO_PREFIX=${HF_REPO_PREFIX} \
        east_scripts/jobs/submit_full_cascade.pbs)
    echo "  east8b x ${variant}: ${east8b_job} (depends on ${data_dep})"

    nilechat_job=$(qsub_id qsub -W depend=afterok:${data_dep}:${STAGE1_JOB} \
        -v PHASE=stage2,MODEL_LABEL=nilechat,VARIANT=${variant},HF_REPO_PREFIX=${HF_REPO_PREFIX},NILECHAT_STAGE1_MODEL_OVERRIDE=${NILECHAT_STAGE1_REPO} \
        east_scripts/jobs/submit_full_cascade.pbs)
    echo "  nilechat x ${variant}: ${nilechat_job} (depends on ${data_dep} and ${STAGE1_JOB})"
done

echo "Done submitting. Track with: qstat -u \$USER"
echo "If any job's dependency never resolves (upstream job failed rather than"
echo "completed 'afterok'), that job stays queued forever -- check qstat -f <jobid>"
echo "and 'qdel' + resubmit once the upstream issue is fixed."
