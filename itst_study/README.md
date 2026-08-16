# ITST across five word orders

Applies **ITST** (Information-Transport-based Simultaneous Translation, Zhang and Feng, EMNLP 2022,
[arXiv:2210.12357](https://arxiv.org/pdf/2210.12357.pdf), code
[ictnlp/ITST](https://github.com/ictnlp/ITST)) to the five languages of
[word_order_study/](../word_order_study/README.md).

**Findings are in [RESULTS.md](RESULTS.md).**

## What ITST does

A simultaneous translation system needs a policy: after each source word, either WRITE the next
target word or READ one more source word. ITST decides this by measuring **how much of the source
information the current target token has already received**.

On top of normal cross-attention it learns a second matrix `T`, where `T_ij` is the share of
information flowing from source token `j` to target token `i`:

- `T_ij = sigmoid(q_info_i . k_info_j)`, from two extra projections
  ([ITST/fairseq/modules/itst_multihead_attention.py:360](ITST/fairseq/modules/itst_multihead_attention.py#L360)).
- Attention is multiplied by `T` and renormalised, so `T` is trained by the ordinary translation
  loss (same file, lines 424-426).
- A **latency loss** pushes `T` towards the diagonal, so a target token does not depend on source
  tokens far ahead of it
  ([label_smoothed_cross_entropy_with_itst_t2t.py:37](ITST/fairseq/criterions/label_smoothed_cross_entropy_with_itst_t2t.py#L37)).
- A **normalisation loss** keeps each row of `T` summing to 1, so "share of information" is
  meaningful (same file, lines 96-101).

At decoding time one number controls the speed/quality trade-off: write token `i` as soon as
`sum_j T_ij >= delta` over the source read so far, otherwise read more. Small `delta` writes early
(low latency), large `delta` waits (high latency). Training anneals a threshold from 1.0 down to
0.5 (`train_threshold = 0.5 + 0.5*exp(-updates/100000)`, same file line 84), so **one** model
serves every latency instead of one model per latency.

## Two tracks, and why

| Track | Data | Vocabulary | Metric | Purpose |
|---|---|---|---|---|
| `paper-envi` | IWSLT15 En-Vi, 131,613 train pairs | word level, freq < 5 to `<unk>` | tokenized BLEU (`multi-bleu.perl`) | Check our pipeline reproduces the ITST paper before trusting it |
| study | The 5 languages, full scale | joint SentencePiece BPE, 10k | detokenized BLEU + chrF++ (sacrebleu) | The actual cross-language question |

The study track keeps `word_order_study`'s **exact dev and test sentences** and removes them from
training, so an ITST score sits next to the chunking-method scores in
[word_order_study/RESULTS.md](../word_order_study/RESULTS.md) without a data mismatch.

**Why the training data is not the word-order study's.** That study fine-tunes an 8B model with
LoRA, where 2,400 pairs is enough. ITST trains a Transformer from scratch, which needs far more.
`source_itst_data.py` re-reads the same Hugging Face sources without subsampling; sizes are in
[data/corpus_sizes.md](data/corpus_sizes.md).

**Corpus size is held constant, on purpose.** The sources are very unequal (Vietnamese 130k, MSA
351k, Korean 162k clean pairs). Training each on everything it has would let a size difference
appear as a word-order effect, so `--max_train` caps them all to the same number, sampled with a
fixed seed rather than sliced off the front (the flat sources are ordered by TED talk, so a slice
would come from a few talks only):

| Group | Datasets | Train pairs each |
|---|---|---:|
| Main word-order comparison | `vietnamese`, `msa`, `korean` | 130,000 |
| Within-Arabic comparison | `saudi-matched`, `egyptian` | 4,323 |

`saudi` (9,650, uncapped) also exists, for a dialect system that uses everything available. It is
not comparable with `egyptian` and is not part of the within-Arabic table.

**Deviation from the paper.** `nlp.stanford.edu` now answers 403, and in the Hugging Face mirror
(`thainq107/iwslt2015-en-vi`) the validation split is a byte-identical copy of the test split, so
tst2012 is not available. Test is still the paper's tst2013 (1,268 pairs), so BLEU stays
comparable; dev is the last 1,553 training pairs, held out of training, and only affects which
checkpoint is picked.

**The two dialects cannot be trained from scratch.** Alexandria SA and EG yield 9,650 and 4,323
pairs in total, against 130k-350k for the other three. Those runs therefore start from the MSA
checkpoint (`INIT_FROM=msa`) and reuse MSA's SentencePiece model and dictionary, so the weights
line up. Any dialect number is a transfer result, not a from-scratch one, and RESULTS.md says so
wherever one appears.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 0. Environment | `setup_itst.sh` | venv at `~/venvs/itst` + the `ITST/` checkout |
| 1. Data | `source_itst_data.py` | `data/<lang>/{train,dev,test}.{en,xx}`, `data/corpus_sizes.md` |
| 2. Encode + binarize | `prepare_fairseq.py` | `data-bin/<dataset>/`, `spm/<lang>/spm.model` |
| 3. Training | `submit_itst_train.pbs` | `checkpoints/<dataset>/` |
| 4a. Threshold sweep | `submit_itst_eval.pbs` -> `eval_itst.py` | `results/<dataset>/pred.delta*.txt` + latency |
| 4b. Scoring | `submit_itst_eval.pbs` -> `score_itst.py` | quality added to `results/<dataset>/results.json` |
| 5. Tables | `summarise_itst.py` | the tables in RESULTS.md |

Steps 4a and 4b are separate because they need **different sacrebleu versions**: fairseq imports
`sacrebleu.tokenizers.TOKENIZERS`, which 2.x removed, so the fairseq env is stuck on 1.5.1; but
chrF++ is `corpus_chrf(..., word_order=2)`, which only exists from 2.x. `setup_itst.sh` therefore
builds a second tiny venv at `~/venvs/itst-score` holding only sacrebleu 2.x, and the eval job
calls it for step 4b. chrF++ is computed with the same call as
`east_scripts/eval/rescore_with_chrf_bertscore.py`, so the numbers are comparable.

Step 1 runs locally (it needs `datasets`, which the fairseq venv does not have); the text files are
then copied to Katana. Steps 2-5 run on Katana inside the ITST venv:

```bash
source itst_study/env.sh
```

`ITST/` is the upstream fairseq fork and is **not** in git; `setup_itst.sh` clones it.

## Setting up

```bash
bash itst_study/setup_itst.sh          # on Katana, from the repo root
```

It pins the oldest stack that still drives Katana's GPUs: python 3.10, torch 2.1.0/cu121,
`numpy<1.24` (fairseq still uses the removed `np.float` alias), `omegaconf==2.0.6` with
`pip==23.3.2` (newer pip rejects that release's metadata).

## Running jobs

Walltime goes on the command line, and the GPU stays unpinned; a job landing on the Blackwell node
dies in about 4 minutes with `cudaErrorNoKernelImageForDevice` and can just be resubmitted.

```bash
QS="-l select=1:ncpus=8:mem=48gb:ngpus=1"

# reproduction
qsub -l walltime=12:00:00 $QS -v DATASET=paper-envi,ARCH=small itst_study/submit_itst_train.pbs
qsub -l walltime=2:00:00  $QS -v DATASET=paper-envi,METRIC=multibleu,THRESHOLDS=0.1:0.2:0.3:0.4:0.5 \
     itst_study/submit_itst_eval.pbs

# the three languages with enough data. Transformer-Small, the same size the paper used for
# En-Vi, because these corpora are the same 130k scale.
for L in vietnamese msa korean; do
  qsub -l walltime=4:00:00 $QS -v DATASET=$L,ARCH=small itst_study/submit_itst_train.pbs; done
for L in vietnamese msa korean; do
  qsub -l walltime=2:00:00 $QS -v DATASET=$L itst_study/submit_itst_eval.pbs; done

# the dialects, starting from MSA (only after the MSA run has finished)
for L in saudi-matched egyptian; do
  qsub -l walltime=2:00:00 $QS -v DATASET=$L,ARCH=small,INIT_FROM=msa,MAX_UPDATE=8000 \
       itst_study/submit_itst_train.pbs; done
for L in saudi-matched egyptian; do
  qsub -l walltime=2:00:00 $QS -v DATASET=$L,SPM_FROM=msa itst_study/submit_itst_eval.pbs; done
```

`THRESHOLDS` is colon separated: `qsub -v` splits its own value on commas and cannot carry spaces.

Add `SMOKE_TEST=1` to a training job for a 300-update run into `$ITST_CKPT/smoke-<dataset>/`.

**Checkpoints go to `/srv/scratch/$USER/itst/checkpoints`** (`ITST_CKPT`, set by `env.sh`), never to
home. One checkpoint is about 520 MB and training keeps ten of them, so a few concurrent runs
exhaust the home quota; when that happened every job died with exit 1 and PBS could not write its
own log back, which makes the cause hard to see.

Decoding is batch-size 1, beam 1, because the policy runs one token at a time. Budget roughly one
second per test sentence per threshold.

## Job names and logs

`itst-train.o<id>` and `itst-eval.o<id>`, in `~/ondemand/EAST`. Poll by job **ID**, never by name:
PBS truncates names in `qstat`.

```bash
qstat -xf <id> | grep -o 'job_state = [A-Z]'          # F means finished
grep -l cudaErrorNoKernelImageForDevice itst-*.o*     # these need resubmitting
```
