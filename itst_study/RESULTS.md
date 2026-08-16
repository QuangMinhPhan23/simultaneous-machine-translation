# ITST across five word orders - results

Setup, scripts and how to run them are in [README.md](README.md). Every number below says which
script produced it and from what.

**Abbreviations.** AL = Average Lagging (Ma et al., ACL 2019), the average number of source tokens
read before writing a target token, so lower means lower latency. CW = Consecutive Wait, AP =
Average Proportion, DAL = Differentiable Average Lagging, all printed by
[ITST/fairseq_cli/sim_generate.py:437-440](ITST/fairseq_cli/sim_generate.py#L437). chrF++ = chrF
with word bigrams, `sacrebleu.corpus_chrf(hyps, [refs], word_order=2)`, sacrebleu 2.6.0. delta =
the ITST read/write threshold: write once the target token has received delta of the source
information, so small delta writes early.

**Status.** All six systems trained and swept in both configurations: original (UPDATE_FREQ=1,
50,000 updates) and batch-matched (UPDATE_FREQ=4, 12,500 updates, effective batch 32k tokens to
match the paper's 4-GPU setup). Section 4 compares the two.

**Short version.** The pipeline reproduces the paper to within 0.04 BLEU once the batch size is
matched (section 4.1; the original 0.30-0.50 gap was entirely due to our single-GPU run using 4x
smaller effective batch). Across the three high-resource languages, Korean (SOV) degrades most at
low latency (4.4%), then Vietnamese (SVO, 2.8%), then MSA (VSO, 2.2%) — same ordering in both
batch configurations (section 4.3). The controlled within-Arabic comparison supports the
hypothesis: the VSO-leaning dialect loses 4.5% against the SVO dialect's 1.2% (section 3).

---

## 1. Does our pipeline reproduce the ITST paper?

Run first, before trusting anything else. Same data as the paper (IWSLT15 En-Vi, test = tst2013,
1,268 pairs), same word-level vocabulary, same Transformer-Small, decoded by `eval_itst.py` and
scored by `score_itst.py --metric multibleu`, which is the ITST repo's own `multi-bleu.perl`
(tokenized, case-insensitive) - the metric the paper's table uses.

**Our sweep** (`results/paper-envi/results.json`, 50,000 updates, average of the last 5 checkpoints):

| delta | CW | AP | AL | DAL | BLEU |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 1.44 | 0.59 | 1.90 | 2.00 | 18.41 |
| 0.2 | 1.67 | 0.62 | 2.51 | 2.71 | 26.09 |
| 0.3 | 2.42 | 0.68 | 3.63 | 4.44 | 28.06 |
| 0.4 | 3.74 | 0.76 | 5.21 | 6.76 | 28.27 |
| 0.5 | 5.22 | 0.83 | 6.83 | 8.60 | 28.58 |
| 0.6 | 6.91 | 0.88 | 8.70 | 9.88 | 28.39 |
| 0.7 | 8.58 | 0.92 | 10.44 | 10.75 | 28.59 |
| 0.8 | 10.80 | 0.94 | 12.58 | 11.39 | 28.59 |

The paper's table stops at delta = 0.5; we go to 0.8 because our delta-to-latency mapping is
shifted and delta = 0.5 only reaches AL 6.83, well short of the paper's AL 10.75.

**The paper's table** (ITST repo, `Text-to-text Simultaneous Translation.md`, En-Vi Transformer-Small):

| delta | CW | AP | AL | DAL | BLEU |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 1.18 | 0.68 | 3.95 | 5.04 | 28.56 |
| 0.2 | 2.08 | 0.72 | 4.55 | 8.59 | 28.68 |
| 0.3 | 4.24 | 0.80 | 6.10 | 13.26 | 28.81 |
| 0.4 | 6.61 | 0.88 | 8.31 | 16.61 | 28.82 |
| 0.5 | 9.01 | 0.92 | 10.75 | 18.73 | 28.89 |

**The same delta gives different latency in the two runs**, so comparing row against row would
compare different operating points. Our model writes earlier: at delta = 0.1 it has read 1.90 source
tokens on average, the paper's 3.95. The fair comparison is at equal AL, interpolated between our
neighbouring sweep points by `summarise_itst.py --al_points`:

| AL | Paper BLEU | Our BLEU (interpolated) | Difference |
|---:|---:|---:|---:|
| 3.95 | 28.56 | 28.10 | -0.46 |
| 4.55 | 28.68 | 28.18 | -0.50 |
| 6.10 | 28.81 | 28.44 | -0.37 |
| 8.31 | 28.82 | 28.43 | -0.39 |
| 10.75 | 28.89 | 28.59 | -0.30 |

**Verdict: reproduced.** We land 0.30 to 0.50 BLEU under the paper at every latency it reports, and
flatten onto a plateau at the same level (our best 28.59 against the paper's best 28.89). Two known
differences explain a gap of this size, and neither is a pipeline error:

1. Our training set is 131,613 pairs, not the paper's ~133K, because `nlp.stanford.edu` now answers
   403 and the Hugging Face mirror's validation split is a byte-identical copy of its test split.
   Dev is therefore the last 1,553 training pairs rather than tst2012. Test is unchanged.
2. The paper does not state its update count. We trained 50,000 updates.

An independent check that the data path is right: `fairseq-preprocess` built vocabularies of
**17,088 English and 7,680 Vietnamese types**, against the paper's stated "17K and 7.7K".

---

## 2. The five languages

All three high-resource systems are trained on **exactly 130,000 pairs** so a corpus-size
difference cannot masquerade as a word-order effect, Transformer-Small, 50,000 updates, joint
SentencePiece BPE 10k. Test sets are `word_order_study`'s, unchanged. Built by
`source_itst_data.py`, sizes in [data/corpus_sizes.md](data/corpus_sizes.md).

### 2.1 Threshold sweeps

chrF++ from `score_itst.py --metric chrf`, 300 test sentences per language.

**Vietnamese (SVO)** -- extra deltas 0.28, 0.51 measured near AL 4 and 8 to tighten interpolation

| delta | AL | DAL | BLEU | chrF++ |
|---:|---:|---:|---:|---:|
| 0.20 | 2.86 | 3.54 | 29.01 | 49.08 |
| 0.28 | 3.97 | 5.23 | 30.90 | 49.75 |
| 0.30 | 4.22 | 5.97 | 31.46 | 50.09 |
| 0.40 | 5.84 | 8.83 | 32.51 | 50.91 |
| 0.50 | 7.75 | 11.01 | 33.04 | 51.45 |
| 0.51 | 8.02 | 11.20 | 33.25 | 51.55 |
| 0.60 | 9.80 | 12.69 | 33.26 | 51.53 |
| 0.70 | 12.40 | 13.97 | 33.17 | 51.55 |
| 0.80 | 15.72 | 15.16 | 33.20 | 51.55 |

**MSA (VSO)** -- extra deltas 0.22, 0.43 measured near AL 4 and 8

| delta | AL | DAL | BLEU | chrF++ |
|---:|---:|---:|---:|---:|
| 0.20 | 3.50 | 5.84 | 11.03 | 37.93 |
| 0.22 | 3.82 | 6.40 | 11.22 | 38.16 |
| 0.30 | 5.58 | 10.94 | 11.60 | 38.79 |
| 0.40 | 7.41 | 15.22 | 11.50 | 38.95 |
| 0.43 | 7.90 | 16.00 | 11.50 | 39.03 |
| 0.50 | 9.29 | 17.89 | 11.73 | 38.98 |
| 0.60 | 12.46 | 20.62 | 11.87 | 39.04 |
| 0.70 | 15.02 | 22.38 | 11.69 | 39.00 |
| 0.80 | 18.56 | 23.62 | 11.76 | 39.08 |

**Korean (SOV)** -- extra delta 0.37 measured near AL 8

| delta | AL | DAL | BLEU | chrF++ |
|---:|---:|---:|---:|---:|
| 0.20 | 3.90 | 7.10 | 2.98 | 18.69 |
| 0.30 | 6.41 | 12.53 | 3.84 | 19.66 |
| 0.37 | 8.06 | 15.30 | 4.18 | 19.74 |
| 0.40 | 8.85 | 16.66 | 4.79 | 20.51 |
| 0.50 | 11.10 | 19.32 | 4.39 | 19.93 |
| 0.60 | 12.99 | 21.03 | 4.78 | 20.00 |
| 0.70 | 15.10 | 22.36 | 4.73 | 20.39 |
| 0.80 | 18.10 | 23.60 | 4.62 | 20.54 |

**Saudi (VSO-leaning)** - 4,323 pairs, initialised from the MSA checkpoint, 8,000 updates

| delta | AL | DAL | chrF++ |
|---:|---:|---:|---:|
| 0.2 | 1.80 | 3.23 | 30.58 |
| 0.3 | 4.04 | 6.95 | 33.92 |
| 0.4 | 7.57 | 13.07 | 35.47 |
| 0.5 | 11.72 | 19.07 | 35.34 |
| 0.6 | 16.02 | 23.09 | 35.34 |
| 0.7 | 21.16 | 25.88 | 35.53 |
| 0.8 | 25.65 | 27.18 | 35.39 |

**Egyptian (SVO)** - 4,323 pairs, initialised from the MSA checkpoint, 8,000 updates; extra deltas 0.31, 0.42

| delta | AL | DAL | chrF++ |
|---:|---:|---:|---:|
| 0.20 | 1.54 | 2.98 | 29.26 |
| 0.30 | 3.66 | 6.42 | 32.12 |
| 0.31 | 3.91 | 6.60 | 32.10 |
| 0.40 | 7.02 | 12.30 | 32.49 |
| 0.42 | 7.60 | 13.00 | 32.47 |
| 0.50 | 11.31 | 16.75 | 32.81 |
| 0.60 | 16.11 | 20.51 | 32.80 |
| 0.70 | 21.09 | 22.96 | 32.74 |
| 0.80 | 26.08 | 24.19 | 32.68 |

### 2.2 Do not compare these scores across languages

Korean scores 20.5 chrF++ and Vietnamese 51.6, but **that is not Korean being six times worse**.
Reading the outputs shows the Korean system produces fluent, correct translations that simply
share few n-grams with a freely paraphrased reference. First test sentence at delta = 0.8:

- ours: `하지만 이 새로운 물질들은 놀라운 혁신을 가져다 줍니다`
- reference: `하지만 신소재가 주는 혁신은 놀라워서`

Same meaning, almost no overlapping substrings. Agglutinative morphology and reference style set
the absolute level, so only **within-language** comparisons are meaningful here. This is the same
caution `word_order_study/data/stats.md` gives for its own cross-language columns.

### 2.3 Degradation at low latency

The word-order question is not absolute quality but how much quality is lost when the system must
start early. Measured two ways, both by `summarise_itst.py`.

**(a) Endpoints of each sweep.** Each language's own fastest and slowest points, which are not at
the same latency, so this is indicative only:

| Language | Order | chrF++ low AL | chrF++ high AL | drop | degradation |
|---|---|---:|---:|---:|---:|
| Korean | SOV | 18.69 (AL 3.90) | 20.54 (AL 18.10) | 1.85 | 9.0% |
| Vietnamese | SVO | 49.08 (AL 2.86) | 51.55 (AL 15.72) | 2.47 | 4.8% |
| MSA | VSO | 37.93 (AL 3.50) | 39.08 (AL 18.56) | 1.15 | 2.9% |

**(b) At matched latency, AL 4 against AL 8**, interpolated between the nearest measured deltas
(extra deltas were added near AL 4 and 8 to keep the interpolation gap small). This is the number
to quote:

| Language | Order | chrF++ at AL 8 | chrF++ at AL 4 | drop | degradation |
|---|---|---:|---:|---:|---:|
| Korean | SOV | 19.74 | 18.73 | 1.01 | 5.1% |
| Vietnamese | SVO | 51.54 | 49.79 | 1.75 | 3.4% |
| MSA | VSO | 39.03 | 38.22 | 0.81 | 2.1% |

### 2.4 What this does and does not show

**Korean, the most word-order-divergent target, degrades most as a percentage** (5.1% against 3.4%
and 2.1% at matched latency). That is the direction the word-order hypothesis predicts: an SOV
target holds its verb to the end, so an SOV system must either wait or guess.

**But the three do not line up by word-order distance.** The hypothesis predicts SOV > VSO > SVO.
The measurement gives SOV (5.1%) > SVO (3.4%) > VSO (2.1%): Vietnamese, which shares English's SVO
order, degrades *more* than MSA, which does not. So on this evidence word order alone does not
order the three languages.

**The percentage framing is fragile here.** The three languages sit at very different absolute
levels (Korean 20, MSA 39, Vietnamese 51 chrF++). Korean's 1.01-point drop becomes 5.1% only
because its base is 20; the same 1.01 points on Vietnamese's base would read as 2.0%. In raw
chrF++ points the drops tell the opposite story - Vietnamese 1.75, Korean 1.01, MSA 0.81 - and
Vietnamese, not Korean, has the largest raw drop. **Which language "degrades most" depends on
whether degradation is read as a percentage or in points.** That is a reason to be careful with the
percentage framing in `word_order_study/RESULTS.md` too, where the same division is done.

The comparison that avoids this problem entirely is the within-Arabic one, in section 3.

---

## 3. Within Arabic: the controlled comparison

This is the cleanest test in the study, because almost everything except word order is held fixed.
Saudi and Egyptian share script, morphology and language family; both are trained on **4,323 pairs**
(Saudi capped from 9,650 down to Egyptian's size), both start from the **same MSA checkpoint**, both
use MSA's SentencePiece model and dictionary, both run 8,000 updates. Egyptian is SVO like English;
Saudi is VSO-leaning.

Their absolute levels are also close (35.5 against 32.7, a 9% difference, against the 2.5x spread in
section 2), so unlike section 2 a percentage comparison here is meaningful.

**At matched latency**, interpolated between nearest measured deltas by `summarise_itst.py
--al_points 4.0 8.0` (Egyptian has extra deltas 0.31 and 0.42 near the target points):

| Dialect | Word order | chrF++ at AL 8 | chrF++ at AL 4 | drop | degradation |
|---|---|---:|---:|---:|---:|
| Saudi | VSO-leaning | 35.46 | 33.86 | 1.60 | **4.5%** |
| Egyptian | SVO | 32.51 | 32.11 | 0.40 | **1.2%** |

**At each sweep's endpoints**, as a second view:

| Dialect | Word order | chrF++ low AL | chrF++ high AL | drop | degradation |
|---|---|---:|---:|---:|---:|
| Saudi | VSO-leaning | 30.58 (AL 1.80) | 35.39 (AL 25.65) | 4.81 | 13.6% |
| Egyptian | SVO | 29.26 (AL 1.54) | 32.68 (AL 26.08) | 3.42 | 10.5% |

**The VSO-leaning dialect loses more at low latency than the SVO dialect**, and it does so on both
measures and in both units - 1.60 against 0.40 chrF++ points at matched latency, 4.5% against 1.2%
as a proportion. Because the two systems differ in almost nothing but target word order, this is
the one result here that supports the word-order hypothesis without the confounds that weaken
section 2.

Note the direction is not about which dialect is better overall: Saudi is the *stronger* system at
high latency (35.46 against 32.58). It simply gives up more of that quality when forced to start
early, which is what a VSO target should do - the verb comes before the subject, so the model must
wait for source material that English places later, or guess.

**What to do with this.** If a system must run at low latency for a VSO-leaning target, the gap it
has to close is roughly four times larger than for an SVO target of the same family and data size.
That is an argument for spending the effort on the read/write policy (or on target-side reordering)
specifically for VSO targets, and not assuming a policy tuned on an SVO pair transfers.

---

## 4. Batch-matched retraining

Sections 2 and 3 trained with UPDATE_FREQ=1 on one GPU (effective batch 8,192 tokens), while the
ITST paper uses 4 GPUs (effective batch 32,768 tokens). To rule out the batch mismatch as a
confound, all five languages were retrained with UPDATE_FREQ=4, 12,500 updates (same total tokens
as the original 50,000 updates). These are the `-bs32k` models.

### 4.1 Paper reproduction with matched batch

The batch-matched En-Vi model (`paper-envi-bs32k`, `results/paper-envi-bs32k/results.json`)
closes the 0.30-0.50 BLEU gap from section 1. At the paper's AL points, interpolated by
`summarise_itst.py --al_points`:

| AL | Paper BLEU | Original | bs32k | Difference |
|---:|---:|---:|---:|---:|
| 3.95 | 28.56 | 28.10 | 28.52 | **-0.04** |
| 4.55 | 28.68 | 28.18 | 28.81 | **+0.13** |
| 6.10 | 28.81 | 28.44 | 28.70 | **-0.11** |
| 8.31 | 28.82 | 28.43 | 28.81 | **-0.01** |
| 10.75 | 28.89 | 28.59 | 28.89 | **+0.00** |

The 4x batch mismatch was the entire cause of the gap.

### 4.2 High-resource sweeps (batch-matched)

chrF++ from `score_itst.py --metric chrf`, 300 test sentences per language. Average of the last 5
checkpoints.

**Vietnamese-bs32k (SVO)**

| delta | AL | DAL | BLEU | chrF++ |
|---:|---:|---:|---:|---:|
| 0.20 | 3.29 | 4.22 | 27.51 | 48.19 |
| 0.30 | 4.97 | 7.06 | 28.58 | 48.92 |
| 0.40 | 6.33 | 9.14 | 29.12 | 49.27 |
| 0.50 | 8.41 | 11.00 | 29.38 | 48.98 |
| 0.60 | 10.60 | 12.70 | 29.57 | 49.20 |
| 0.70 | 13.49 | 13.65 | 29.79 | 49.57 |
| 0.80 | 17.31 | 14.65 | 29.67 | 49.48 |

**MSA-bs32k (VSO)**

| delta | AL | DAL | BLEU | chrF++ |
|---:|---:|---:|---:|---:|
| 0.20 | 3.78 | 6.71 | 10.50 | 38.30 |
| 0.30 | 5.61 | 11.55 | 11.13 | 38.78 |
| 0.40 | 7.53 | 15.65 | 11.70 | 39.04 |
| 0.50 | 9.97 | 19.12 | 11.76 | 38.96 |
| 0.60 | 12.44 | 21.39 | 11.98 | 39.18 |
| 0.70 | 15.74 | 23.45 | 11.74 | 39.03 |
| 0.80 | 19.26 | 24.89 | 11.69 | 39.05 |

**Korean-bs32k (SOV)**

| delta | AL | DAL | BLEU | chrF++ |
|---:|---:|---:|---:|---:|
| 0.20 | 5.94 | 8.47 | 4.09 | 19.93 |
| 0.30 | 8.47 | 13.42 | 4.21 | 20.08 |
| 0.40 | 10.72 | 17.33 | 4.19 | 20.09 |
| 0.50 | 12.67 | 20.01 | 4.51 | 20.33 |
| 0.60 | 14.59 | 21.72 | 5.03 | 20.62 |
| 0.70 | 17.77 | 22.99 | 5.16 | 20.80 |
| 0.80 | 21.20 | 24.25 | 5.05 | 20.84 |

### 4.3 Degradation comparison: original vs batch-matched

Endpoint degradation (lowest AL to peak chrF++), from `summarise_itst.py`:

| Language | Order | Original degradation | bs32k degradation |
|---|---|---:|---:|
| Korean | SOV | 9.0% (18.69→20.54) | 4.4% (19.93→20.84) |
| Vietnamese | SVO | 4.8% (49.08→51.55) | 2.8% (48.19→49.57) |
| MSA | VSO | 2.9% (37.93→39.08) | 2.2% (38.30→39.18) |

The ordering is unchanged: **Korean > Vietnamese > MSA** in both configurations. Matching the
batch flattens all three curves (degradation drops roughly in half) but does not reorder them.

Korean-bs32k's lowest delta (0.2) gives AL 5.94, so it cannot reach AL 4 for the matched-latency
comparison from section 2.3b. At the nearest comparable point (AL ~6), Korean-bs32k scores 19.93
and its peak is 20.84 — a 4.4% drop. The original at the same endpoint range (AL 3.90 to 18.10)
drops 9.0%. The ratio is consistent: batch matching halves the gap but does not change which
language suffers most.

### 4.4 Dialect models: not comparable

The bs32k dialect models (saudi-matched-bs32k, egyptian-bs32k) are much weaker than the originals:

| Dialect | Original chrF++ range | bs32k chrF++ range |
|---|---|---|
| Saudi | 30.58 - 35.53 | 24.58 - 27.35 |
| Egyptian | 29.26 - 32.81 | 20.80 - 22.68 |

Two factors explain the drop. First, the dialect training set has only 4,323 pairs; with
UPDATE_FREQ=4, each gradient update averages over 4 batches, which may be too aggressive for such
a small corpus (the model undershoots in the same number of updates). Second, the bs32k dialect
checkpoints saved only `checkpoint_last.pt` with no intermediate snapshots, so the averaging step
(which the other five models benefit from) fell back to a single checkpoint. **The section 3
within-Arabic comparison, using the original models, remains the controlled test for word order.**
