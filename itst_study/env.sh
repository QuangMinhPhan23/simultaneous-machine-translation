# Activate the ITST fairseq environment. Source this, do not run it.
#   source itst_study/env.sh
module purge
module load python/3.10.8
source "$HOME/venvs/itst/bin/activate"
export ITST_REPO="$HOME/ondemand/EAST/itst_study/ITST"
export PYTHONPATH="$ITST_REPO:${PYTHONPATH:-}"

# Checkpoints live on scratch, never in home. One fairseq checkpoint is about 520 MB and training
# keeps ten, so a few concurrent runs blow the small home quota; when that happened the jobs died
# with exit 1 and PBS could not even write its own log back.
export ITST_CKPT="/srv/scratch/$USER/itst/checkpoints"
mkdir -p "$ITST_CKPT"
