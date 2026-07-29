# Source this (not `bash env.sh`!) at the start of every Katana terminal
# session, before running any python command directly: `source east_scripts/env.sh`
#
# `export` statements inside a script run as `bash script.sh` only affect that
# child process, not your interactive shell -- so running setup_katana_env.sh
# or run_*.sh (which each redo these exports internally for their own
# subprocess) does NOT leave HF_HOME etc. set in your shell afterward. Any
# python command you then run directly (not through one of those scripts)
# silently falls back to the default ~/.cache/huggingface on the small home
# quota. Sourcing this file fixes that for the current shell.
export EAST_CACHE=${EAST_CACHE:-${TMPDIR:-/scratch/$USER}/east_cache}
mkdir -p "$EAST_CACHE/huggingface" "$EAST_CACHE/torch" "$EAST_CACHE/pip"
export HF_HOME="$EAST_CACHE/huggingface"
export TRANSFORMERS_CACHE="$EAST_CACHE/huggingface"
export HF_HUB_CACHE="$EAST_CACHE/huggingface/hub"
export TORCH_HOME="$EAST_CACHE/torch"
# pip's download/wheel cache defaults to ~/.cache/pip regardless of where the
# venv itself lives -- redirect it too, since it alone has filled the home
# quota (6.7GB) on its own in past sessions.
export PIP_CACHE_DIR="$EAST_CACHE/pip"
echo "Caches set to $EAST_CACHE for this shell."
