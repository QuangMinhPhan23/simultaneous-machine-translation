# ITST across five word orders

**ITST** (Information-Transport-based Simultaneous Translation, Zhang & Feng, EMNLP 2022,
[arXiv:2210.12357](https://arxiv.org/pdf/2210.12357.pdf),
[ictnlp/ITST](https://github.com/ictnlp/ITST)) applied to the five languages from
[word_order_study/](../word_order_study/README.md).

**Results in [RESULTS.md](RESULTS.md).**

## How ITST works

One trained model serves all latencies. At decode time, a threshold `delta` controls when to write:
small delta = write early (low latency), large delta = wait (high latency). The model learns an
information-transport matrix `T` on top of cross-attention, and writes target token `i` once
`sum_j T_ij >= delta`.

## Two tracks

| Track | Data | Vocab | Metric | Purpose |
|---|---|---|---|---|
| `paper-envi` | IWSLT15 En-Vi, 131k pairs | word level | tokenized BLEU | Reproduce the paper |
| study | 5 languages, 130k each | SentencePiece BPE 10k | chrF++ | Cross-language comparison |

Dev and test are the same sentences as `word_order_study/`.

| Group | Datasets | Train pairs |
|---|---|---:|
| High-resource | `vietnamese`, `msa`, `korean` | 130,000 each |
| Dialect (transfer from MSA) | `saudi-matched`, `egyptian` | 4,323 each |

Corpus sizes capped so size differences cannot look like word-order effects. Sizes in
[data/corpus_sizes.md](data/corpus_sizes.md).

Dialects start from the MSA checkpoint (`INIT_FROM=msa`) and reuse MSA's SPM model and dictionary.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 0. Environment | `setup_itst.sh` | `~/venvs/itst` + `ITST/` checkout |
| 1. Data | `source_itst_data.py` | `data/<lang>/{train,dev,test}.{en,xx}` |
| 2. Encode + binarize | `prepare_fairseq.py` | `data-bin/<dataset>/`, `spm/<lang>/spm.model` |
| 3. Train | `submit_itst_train.pbs` | `checkpoints/<dataset>/` |
| 4a. Decode | `submit_itst_eval.pbs` -> `eval_itst.py` | `results/<dataset>/pred.delta*.txt` |
| 4b. Score | `submit_itst_eval.pbs` -> `score_itst.py` | `results/<dataset>/results.json` |
| 5. Tables | `summarise_itst.py` | tables for RESULTS.md |

Steps 4a/4b use different sacrebleu versions (fairseq needs 1.x, chrF++ needs 2.x), so
`setup_itst.sh` builds a second venv at `~/venvs/itst-score`.

Step 1 runs locally; steps 2-5 on Katana: `source itst_study/env.sh`

## Setup

```bash
bash itst_study/setup_itst.sh    # on Katana, from repo root
```

## Running jobs

```bash
QS="-l select=1:ncpus=8:mem=48gb:ngpus=1"

# paper reproduction
qsub -l walltime=12:00:00 $QS -v DATASET=paper-envi,ARCH=small itst_study/submit_itst_train.pbs
qsub -l walltime=2:00:00  $QS -v DATASET=paper-envi,METRIC=multibleu,THRESHOLDS=0.1:0.2:0.3:0.4:0.5 \
     itst_study/submit_itst_eval.pbs

# high-resource languages
for L in vietnamese msa korean; do
  qsub -l walltime=4:00:00 $QS -v DATASET=$L,ARCH=small itst_study/submit_itst_train.pbs; done
for L in vietnamese msa korean; do
  qsub -l walltime=2:00:00 $QS -v DATASET=$L itst_study/submit_itst_eval.pbs; done

# dialects (after MSA finishes)
for L in saudi-matched egyptian; do
  qsub -l walltime=2:00:00 $QS -v DATASET=$L,ARCH=small,INIT_FROM=msa,MAX_UPDATE=8000 \
       itst_study/submit_itst_train.pbs; done
for L in saudi-matched egyptian; do
  qsub -l walltime=2:00:00 $QS -v DATASET=$L,SPM_FROM=msa itst_study/submit_itst_eval.pbs; done
```

Checkpoints go to `/srv/scratch/$USER/itst/checkpoints` (`ITST_CKPT` in `env.sh`), not home.

Logs: `itst-train.o<id>`, `itst-eval.o<id>` in `~/ondemand/EAST`. Poll by job ID, not name.
