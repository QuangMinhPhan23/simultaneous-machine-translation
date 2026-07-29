#!/bin/bash
# Run once inside a Katana JupyterLab terminal (GPU node) before evaluating EAST-8B
# on the Alexandria Arabic test set. Deliberately skips `pip install -e ./` and the
# repo's top-level requirements.txt (the original authors' full training env --
# TensorFlow, Ray, vLLM, WandB, flash-attn's CUDA build -- none of which is needed
# just to load a checkpoint and run inference with simuleval_standalone.py).
#
# Home directories on Katana have a small quota; EAST-8B (~16GB) plus COMET's
# checkpoint will not fit there. Everything goes to $TMPDIR (Katana's per-job
# scratch path under /scratch, set by PBS) instead -- note this is wiped when
# the current JupyterLab job/session ends, so models will need re-downloading
# next session. Before running this:
#   module load python/3.11.3            # torch/transformers/comet wheels
#                                         # target <=3.11, not the 3.13.2 module
#   python3 -m venv "$TMPDIR/east-venv"
#   source "$TMPDIR/east-venv/bin/activate"
set -e

cd "$(dirname "$0")/.."

export EAST_CACHE=${EAST_CACHE:-${TMPDIR:-/scratch/$USER}/east_cache}
mkdir -p "$EAST_CACHE/huggingface" "$EAST_CACHE/torch" "$EAST_CACHE/pip"
export HF_HOME="$EAST_CACHE/huggingface"
export TRANSFORMERS_CACHE="$EAST_CACHE/huggingface"
export HF_HUB_CACHE="$EAST_CACHE/huggingface/hub"
export TORCH_HOME="$EAST_CACHE/torch"
# pip's download/wheel cache defaults to ~/.cache/pip regardless of where the
# venv lives -- redirect it too, it can fill the home quota on its own (6.7GB
# in one observed session).
export PIP_CACHE_DIR="$EAST_CACHE/pip"
echo "Caches redirected to $EAST_CACHE. IMPORTANT: these exports only apply"
echo "inside this script's own process -- 'source east_scripts/env.sh' at the"
echo "start of every new terminal/session before running any python command"
echo "directly, or it'll silently fall back to the (small) home directory cache."

python -m pip install --upgrade pip

# Installed as its own step (not via requirements_eval.txt) to avoid pip's
# index-resolution ambiguity between this pinned build and default PyPI --
# see requirements_eval.txt for why the cu126 index specifically.
#
# Pinned to an exact version (not just "torch", latest-compatible) -- an
# unpinned install let pip's resolver pick a different torch release per job,
# and one job (2026-07-09) hit a genuine ResolutionImpossible backtracking
# failure across several 2.6-2.10 candidates' conflicting pinned sub-deps,
# while other jobs running the identical script around the same time landed
# on a different, working version. 2.12.1+cu126 is pinned here because it's
# the version multiple prior successful jobs already verified working: same
# install every time, no more resolver roulette. Bump deliberately (and
# re-verify) if a newer torch is ever actually needed.
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.12.1+cu126
pip install -r east_scripts/requirements_eval.txt

# BLEURT is intentionally NOT installed here: it pulls in TensorFlow, which
# forces protobuf>=7, conflicting with unbabel-comet's protobuf<5 requirement.
# Pass INSTALL_BLEURT=1 to this script if you want it anyway and are prepared
# to sort out the protobuf clash yourself.
if [ "${INSTALL_BLEURT:-0}" = "1" ]; then
    pip install git+https://github.com/google-research/bleurt.git
    mkdir -p "$EAST_CACHE/checkpoints"
    if [ ! -d "$EAST_CACHE/checkpoints/BLEURT-20" ]; then
        curl -L -o "$EAST_CACHE/checkpoints/BLEURT-20.zip" https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
        unzip -q "$EAST_CACHE/checkpoints/BLEURT-20.zip" -d "$EAST_CACHE/checkpoints"
    fi
fi

# Pre-fetch the COMET checkpoint (lands under $HF_HOME, i.e. $TMPDIR/scratch).
# Note: comet.download_model() takes a full HF Hub repo id, not the bare
# metric name -- "wmt22-comet-da" alone 404s; it must be "Unbabel/wmt22-comet-da".
python - <<'PY'
from comet import download_model
path = download_model("Unbabel/wmt22-comet-da")
print(f"COMET checkpoint downloaded to: {path}")
PY

echo "Setup complete. EAST-8B itself will auto-download from the HF Hub on first use."
