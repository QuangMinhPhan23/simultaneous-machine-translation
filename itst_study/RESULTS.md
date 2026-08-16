# ITST results

Setup and scripts: [README.md](README.md).

AL = Average Lagging (source tokens read before writing, lower = faster).
chrF++ = `sacrebleu.corpus_chrf(hyps, [refs], word_order=2)`.
delta = ITST read/write threshold (small = write early).

All six systems trained in both configurations: original (1-GPU 8k batch) and batch-matched
(1-GPU with UPDATE_FREQ=4, 32k effective batch matching the paper's 4-GPU setup).

---

## 1. Paper reproduction

IWSLT15 En-Vi, 1,268 test pairs, tokenized BLEU via `multi-bleu.perl`.

**Our sweep** (50k updates, last-5 checkpoint average):

| delta | AL | BLEU |
|---:|---:|---:|
| 0.1 | 1.90 | 18.41 |
| 0.2 | 2.51 | 26.09 |
| 0.3 | 3.63 | 28.06 |
| 0.4 | 5.21 | 28.27 |
| 0.5 | 6.83 | 28.58 |
| 0.7 | 10.44 | 28.59 |

**Comparison at matched AL** (interpolated by `summarise_itst.py`):

| AL | Paper | Ours (8k) | Ours (32k) |
|---:|---:|---:|---:|
| 3.95 | 28.56 | 28.10 | 28.52 |
| 6.10 | 28.81 | 28.44 | 28.70 |
| 10.75 | 28.89 | 28.59 | 28.89 |

The 8k-to-32k batch mismatch explains the entire 0.30-0.50 gap.

---

## 2. Five-language sweeps

130k train pairs each, Transformer-Small, 50k updates, chrF++ on 300 test sentences.

**Vietnamese (SVO)**

| delta | AL | chrF++ |
|---:|---:|---:|
| 0.20 | 2.86 | 49.08 |
| 0.30 | 4.22 | 50.09 |
| 0.50 | 7.75 | 51.45 |
| 0.80 | 15.72 | 51.55 |

**MSA (VSO)**

| delta | AL | chrF++ |
|---:|---:|---:|
| 0.20 | 3.50 | 37.93 |
| 0.30 | 5.58 | 38.79 |
| 0.50 | 9.29 | 38.98 |
| 0.80 | 18.56 | 39.08 |

**Korean (SOV)**

| delta | AL | chrF++ |
|---:|---:|---:|
| 0.20 | 3.90 | 18.69 |
| 0.30 | 6.41 | 19.66 |
| 0.50 | 11.10 | 19.93 |
| 0.80 | 18.10 | 20.54 |

**Saudi (VSO-leaning)** -- 4,323 pairs, from MSA checkpoint, 8k updates

| delta | AL | chrF++ |
|---:|---:|---:|
| 0.20 | 1.80 | 30.58 |
| 0.30 | 4.04 | 33.92 |
| 0.50 | 11.72 | 35.34 |
| 0.80 | 25.65 | 35.39 |

**Egyptian (SVO)** -- 4,323 pairs, from MSA checkpoint, 8k updates

| delta | AL | chrF++ |
|---:|---:|---:|
| 0.20 | 1.54 | 29.26 |
| 0.30 | 3.66 | 32.12 |
| 0.50 | 11.31 | 32.81 |
| 0.80 | 26.08 | 32.68 |

Cross-language absolute scores are not comparable (Korean 20 vs Vietnamese 51 is morphology and
reference style, not quality). Only within-language degradation is meaningful.

### Degradation at matched latency

Interpolated at AL 4 and AL 8 by `summarise_itst.py --al_points 4.0 8.0`:

| Language | Order | chrF++ AL=8 | chrF++ AL=4 | degradation |
|---|---|---:|---:|---:|
| Korean | SOV | 19.74 | 18.73 | **5.1%** |
| Vietnamese | SVO | 51.54 | 49.79 | **3.4%** |
| MSA | VSO | 39.03 | 38.22 | **2.1%** |

Korean (SOV) degrades most, as the word-order hypothesis predicts. But Vietnamese (SVO) degrades
more than MSA (VSO), so word order alone does not fully explain the ranking.

---

## 3. Within-Arabic comparison (controlled)

The cleanest test: Saudi vs Egyptian share script, morphology, language family, same 4,323 train
pairs, same MSA checkpoint, same SPM, same 8k updates. Only word order differs (VSO-leaning vs SVO).

**At matched latency** (AL 4 vs AL 8, `summarise_itst.py`):

| Dialect | Order | chrF++ AL=8 | chrF++ AL=4 | degradation |
|---|---|---:|---:|---:|
| Saudi | VSO-leaning | 35.46 | 33.86 | **4.5%** |
| Egyptian | SVO | 32.51 | 32.11 | **1.2%** |

VSO-leaning loses 4x more at low latency than SVO, with nearly everything else held constant.

---

## 4. Batch-matched (32k effective)

All languages retrained with UPDATE_FREQ=4, 12,500 updates (same total tokens as 50k at 8k batch).

### Degradation: 8k vs 32k

| Language | Order | 8k degradation | 32k degradation |
|---|---|---:|---:|
| Korean | SOV | 9.0% (18.69->20.54) | 4.4% (19.93->20.84) |
| Vietnamese | SVO | 4.8% (49.08->51.55) | 2.8% (48.19->49.57) |
| MSA | VSO | 2.9% (37.93->39.08) | 2.2% (38.30->39.18) |

Same ordering in both: **Korean > Vietnamese > MSA**. Larger batch flattens curves but does not
reorder them.

### Dialect 32k models

Much weaker than originals (Saudi 24-27 vs 30-35, Egyptian 20-23 vs 29-33) due to tiny corpus +
no checkpoint averaging. Section 3's 8k numbers remain the controlled comparison.
