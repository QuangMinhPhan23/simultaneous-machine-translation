#!/bin/bash
# Build the ITST environment on Katana.
# Pins python 3.10 + torch 2.1.0/cu121 + numpy<1.24 to work with both
# the old fairseq fork and Katana's GPUs (L40S sm_89, H200 sm_90).
#
# Usage: bash itst_study/setup_itst.sh
set -euo pipefail

VENV=${VENV:-$HOME/venvs/itst}
ITST_DIR=${ITST_DIR:-$HOME/ondemand/EAST/itst_study/ITST}

# Step 1: python 3.10 from the module system, no conda on this cluster.
module purge
module load python/3.10.8

# Step 2: venv. Set FRESH=1 to rebuild it from scratch.
if [ "${FRESH:-0}" = "1" ]; then rm -rf "$VENV"; fi
[ -d "$VENV" ] || python -m venv "$VENV"
source "$VENV/bin/activate"
# pip<24.1: newer pip rejects omegaconf 2.0.6's metadata.
pip install --upgrade "pip==23.3.2" "setuptools<70" wheel

# Step 3: torch first. torchaudio needed because fairseq imports speech tasks at startup.
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Step 4: fairseq's pinned deps.
pip install "numpy<1.24" "Cython<3" "omegaconf==2.0.6" "hydra-core==1.0.7" \
            "antlr4-python3-runtime==4.8" "sacrebleu==1.5.1" "transformers==4.36.2" \
            bitarray editdistance regex tqdm portalocker soundfile sentencepiece \
            sacremoses subword-nmt psutil requests tensorboardX

# Step 5: get the fork if it is not already here.
if [ ! -d "$ITST_DIR" ]; then
  git clone --depth 1 https://github.com/ictnlp/ITST.git "$ITST_DIR"
  rm -rf "$ITST_DIR/.git"
fi

# Step 6: editable install. No build isolation so it sees torch; compat mode for old setup.py.
cd "$ITST_DIR"
pip install --editable ./ --no-deps --no-build-isolation --config-settings editable_mode=compat

# Step 6b: patch sim_generate.py with a cli_main() entry point (upstream has none).
if ! grep -q "^def cli_main" "$ITST_DIR/fairseq_cli/sim_generate.py"; then
  cat >> "$ITST_DIR/fairseq_cli/sim_generate.py" <<'PY'


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
PY
  echo "patched: added cli_main() to fairseq_cli/sim_generate.py"
fi

# Step 7: check the pieces that matter actually import.
python - <<'PY'
import torch, fairseq
print("torch", torch.__version__, "cuda", torch.version.cuda)
import fairseq.models.transformer_itst
import fairseq.criterions.label_smoothed_cross_entropy_with_itst_t2t
import fairseq.modules.itst_multihead_attention
from fairseq.models import ARCH_MODEL_REGISTRY
from fairseq.criterions import CRITERION_REGISTRY
assert "transformer_itst" in ARCH_MODEL_REGISTRY, "arch transformer_itst not registered"
assert "label_smoothed_cross_entropy_with_itst_t2t" in CRITERION_REGISTRY, "itst criterion missing"
print("fairseq", fairseq.__version__, "- transformer_itst and itst t2t criterion registered")
PY

echo "OK: ITST env ready at $VENV"

# Step 8: separate scoring venv (sacrebleu 2.x for chrF++; fairseq needs 1.x).
SCORE_VENV=${SCORE_VENV:-$HOME/venvs/itst-score}
if [ "${FRESH:-0}" = "1" ]; then rm -rf "$SCORE_VENV"; fi
[ -d "$SCORE_VENV" ] || python -m venv "$SCORE_VENV"
"$SCORE_VENV/bin/pip" install --quiet --upgrade pip
"$SCORE_VENV/bin/pip" install --quiet "sacrebleu>=2.4"
"$SCORE_VENV/bin/python" -c "
import sacrebleu, inspect
assert 'word_order' in inspect.signature(sacrebleu.corpus_chrf).parameters
print('sacrebleu', sacrebleu.__version__, '- chrF++ available')
"
echo "OK: scoring env ready at $SCORE_VENV"
