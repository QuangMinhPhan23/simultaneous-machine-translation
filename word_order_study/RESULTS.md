# Word-order study - Results

Question and setup are in [README.md](README.md). Prediction (least → most quality loss at low latency):
Vietnamese < Egyptian < Saudi < MSA < Korean. Below are the results so far.

## Data built (Step 1)

Five languages, same pipeline: English side filtered to 15-30 words, deduped, badly-aligned pairs
dropped. Split sizes 2400/300/300 (Vietnamese, MSA, Korean) and up to 3000/500/500 (Saudi, Egyptian).
Full per-split table in [data/stats.md](data/stats.md).

## Result 1 - the tokenizer is a latency confound (Step 2)

Average Lagging is often measured in tokens, but the Llama-3 tokenizer splits each language differently,
so equal content becomes a different token count:

| Language | chars/token | tokens/sentence | AL-inflation vs lowest |
|---|---:|---:|---:|
| Vietnamese | 3.66 | 28.7 | 1.00x |
| Saudi | 2.48 | 33.5 | 1.17x |
| Korean | 1.68 | 35.4 | 1.23x |
| Egyptian | 2.42 | 36.4 | 1.27x |
| MSA | 2.55 | 37.6 | 1.31x |

Vietnamese emits ~25% fewer target tokens for the same English, so a token-based AL would *understate*
its latency - dangerous, since Vietnamese is predicted to be the easiest. **Decision:** report latency in
**words (and characters)**, never tokens alone.

## Result 2 - chunker fallback rate is weak evidence (Step 3)

Each sentence is split into aligned read/write chunks at three latencies; when the LLM's split can't
rebuild the sentence, we fall back to a word-count split and record it. Example (low latency):

| Read (English) | Write (Vietnamese) |
|---|---|
| Good morning, | Chào buổi sáng, |
| I need | tôi cần |
| ten kilograms of tomatoes | mười ki-lô cà chua |
| for the restaurant. | cho nhà hàng. |

After fixing a punctuation-only validation bug, the fallback rate stopped tracking word order - it mostly
reflects how faithfully the model copies each script (Vietnamese fell back *more* than Korean, the
opposite of the hypothesis). **So the fallback rate is not good evidence for the hypothesis; the real
test is the quality-vs-latency curve below.**

## Result 2b - the chunks themselves were bad in two measurable ways

**Words cut in half.** Vietnamese whitespace separates syllables, not words, so a word-count split can
end a step inside a word: `dan chu` (democracy) becomes `nen dan` | `chu`.

> **Measured** by `check_word_splits.py` over all 2400 sentences of `data/vietnamese/chunks-generic.json`
> at low latency. Method: segment each target sentence with pyvi, record the syllable index where every
> word ends, then check each chunk boundary lands on one of those indices. Result: **321/395 (81.3%) of
> fallback sentences** and **619/2400 (25.8%) of all sentences** have a boundary inside a word. The
> difference (298 sentences) was chunked by the LLM and accepted by the validator, so the model splits
> Vietnamese words too and a better prompt would not fix it.

**Steps that do not correspond.** The validator only checks that the steps rebuild the sentence and that
both sides have the same count, never that step i means the same as step i, so a segmentation shifted by
one position still passes.

> **Measured** by `check_alignment.py`, output in `data/alignment_check.md`. Method: embed each English
> step and each target step with `paraphrase-multilingual-MiniLM-L12-v2`, take the mean cosine similarity
> of step i to step i (`matched`), then rotate the target steps by one position for a control
> (`shuffled`). 600 step pairs per cell. At low latency the gap `matched - shuffled` was **+0.376
> Vietnamese, +0.416 MSA, +0.316 Saudi, +0.259 Egyptian, +0.150 Korean**, and fallback steps scored 2 to
> 3 times lower than LLM steps in every language.
>
> Two caveats. Short steps are all vaguely similar, which raises the shuffled control at low latency, so
> high-vs-low comparisons overstate the drop; compare LLM against fallback at the *same* latency instead.
> And the encoder handles dialectal Arabic worse than MSA, so this metric's cross-language ordering is
> not trustworthy on its own.

## Result 3 - training (Step 4, done)

10 systems (5 languages x 2 chunking methods), LoRA Stage-II fine-tuning on the EAST-8B base. Each
adapter is pushed to Hugging Face as `Henry236/east8b-<lang>-lora-<method>`.

**All 10 are trained, and each adapter file was checked on the Hub.** A run takes 20 to 30 minutes on
one A100 or L40S GPU, so training is not the bottleneck; waiting for a free GPU on the shared cluster
is. Only the LoRA adapter is saved (about 670 MB) rather than the full merged model (about 16 GB).

## Result 4 - the main test (Step 5): the hypothesis is only half right

> **Measured** by `east_scripts/eval/simuleval_standalone.py`, one job per cell, 30 cells
> (5 languages x 2 methods x 3 latencies). Each run loads the frozen `biaofu-xmu/EAST-8B` base plus that
> system's LoRA adapter from the Hub, decodes the held-out test split (300 sentences for Vietnamese, MSA
> and Korean; 500 for Saudi and Egyptian) with beam size 5, and scores with sacrebleu. COMET is skipped
> because the Alexandria authors report it is unreliable for Arabic dialects. Degradation is
> (chrF_high - chrF_low) / chrF_high. Raw scores are under `results/wordorder/<lang>-<method>/<latency>/`.

Generic chunking:

| Language | Word order | chrF_low | chrF_high | Degradation |
|---|---|---:|---:|---:|
| MSA | VSO | 36.84 | 40.05 | **8.0%** |
| Saudi | VSO-leaning | 38.61 | 42.19 | **8.5%** |
| Egyptian | SVO | 33.60 | 37.17 | **9.6%** |
| Vietnamese | SVO | 42.37 | 49.61 | **14.6%** |
| Korean | SOV | 17.12 | 22.24 | **23.0%** |

**Supported:** Korean (SOV) degrades far more than everything else, 23% against 8-15%. This is the
sharpest prediction of the hypothesis and it holds.

**Not supported:** Vietnamese was predicted to be the easiest and came second worst, degrading more than
all three Arabic varieties. The within-Arabic ordering also came out backwards: predicted
Egyptian (SVO) < Saudi < MSA (VSO), measured MSA < Saudi < Egyptian.

So "more word-order divergence means more degradation" is too simple. It describes Korean well and the
other four badly.

**Most likely confound, and it is testable.** Vietnamese had by far the worst chunk corruption (25.8% of
sentences with a word cut in half, concentrated at low latency, which is exactly where degradation is
measured). Retraining Vietnamese on Method C chunks (0% corruption) makes a clear prediction: if the
corruption caused it, degradation should drop. If it does not, Vietnamese genuinely degrades more than
VSO Arabic and the hypothesis needs rethinking.

**Word-order-aware prompting mostly hurt.** Method B was worse than Method A in 4 of 5 languages
(Vietnamese 14.6 -> 17.0, MSA 8.0 -> 10.4, Egyptian 9.6 -> 14.1, Saudi 8.5 -> 9.9) and helped only
Korean (23.0 -> 20.7), the one language where reordering is severe enough to be worth the extra
instruction.

## Result 5 - most of the apparent word-order effect was a chunking artifact

> **Measured** the same way as Result 4. Method C chunks were built by `align_chunks.py`, turned into
> training data by `build_simt_sft.py --method aligned` (9600 examples per language, 0 fallbacks), and
> trained with `submit_train.pbs` into `Henry236/east8b-<lang>-lora-aligned`. Then 9 more eval cells
> (3 languages x 3 latencies). Same base model, same test sets, same metric: only the chunking changed.

| Language | Word order | A (generic) | B (specific) | **C (aligned)** |
|---|---|---:|---:|---:|
| MSA | VSO | 8.0% | 10.4% | **6.3%** |
| Vietnamese | SVO | 14.6% | 17.0% | **7.0%** |
| Egyptian | SVO | 9.6% | 14.1% | **7.9%** |
| Saudi | VSO-leaning | 8.5% | 9.9% | **8.1%** |
| Korean | SOV | 23.0% | 20.7% | **8.3%** |

Method C is better than both prompt-based methods in **all five languages**. It halves Vietnamese
degradation and cuts Korean's by nearly two thirds.

**Removing wasted steps helps again.** Alignment rarely links punctuation to a source word, so a comma
or full stop often ends up alone in its own step and the model learns to spend a write step emitting
".". Measured by `check_word_splits.py`: 7.9% of Vietnamese low-latency steps were like this (1754 of 22300),
against 0.5% for MSA and 0.1% for Korean. `build_simt_sft.py --merge_punct_only` folds those steps into
the previous one, which removes all 1754 while keeping the text identical and barely touching
granularity (9.29 to 8.56 steps per sentence). Retraining Vietnamese on that data:

| Vietnamese | Degradation |
|---|---:|
| A (generic) | 14.6% |
| B (specific) | 17.0% |
| C (aligned) | 7.0% |
| **C + punctuation merge** | **5.3%** |

A 64% total reduction from the original chunking, with the same base model, recipe and test set.

**Applied to all five, the merge turns out to be a Vietnamese-specific fix, as predicted.** The
prediction was recorded before the runs: it should only matter where punctuation-only steps are common.

| Language | punct-only steps | C aligned | C + merge | change |
|---|---:|---:|---:|---:|
| Vietnamese | 7.9% | 7.0% | 5.3% | **-1.7** |
| Saudi | - | 8.1% | 7.5% | -0.6 |
| Korean | 0.1% | 8.3% | 7.8% | -0.5 |
| Egyptian | - | 7.9% | 8.3% | +0.4 |
| MSA | 0.5% | 6.3% | 7.6% | +1.3 |

Only Vietnamese moves beyond the +/-1.3 point spread of the others, and it is the only language with a
large number of wasted steps to remove. Report this as a targeted fix for languages whose segmenter
strands punctuation, not as a general method.

**A second metric agrees.** `rescore_kiwi.py` re-scored every prediction with COMET-KIWI, which is
reference-free: Vietnamese 17.1% -> 5.9%, Korean 17.4% -> 6.8%, MSA 9.9% -> 7.4% (generic -> aligned).
A completely different kind of metric reaching the same conclusion is much stronger evidence than
chrF++ alone.

**COMET-KIWI also shows the chrF++ numbers were misleading about Korean.** At high latency Korean scores
82.9 KIWI against Vietnamese 81.5, that is, slightly *better*, while chrF++ said 22.2 against 49.6.
Korean's low chrF++ was the free TED references plus its morphology, not model quality. Do not read the
absolute chrF++ gap between languages as a quality ranking.

**The gain is where it should be.** Low-latency quality improves (chrF++ 42.37 -> 45.96 Vietnamese,
17.12 -> 19.95 Korean, 36.84 -> 37.35 MSA) while high-latency quality is essentially unchanged
(49.61 -> 49.42, 22.24 -> 21.76, 40.05 -> 39.87). Better chunks do not make the model better in general,
they make it better when it has to commit early, which is exactly what the chunks teach.

**What this does to the hypothesis.** With the old chunking the spread was 8% to 23%. With alignment
chunking it is 6.3% to 8.3%, so the size of the effect drops by roughly 85%. Most of what looked like
word-order divergence was data quality.

Korean (SOV) is still the worst, which is the study's sharpest prediction and the one part that keeps
surviving. But the rest of the ordering does not hold: MSA (VSO) is the *best*, and within Arabic the
measured order is MSA 6.3% < Egyptian 7.9% < Saudi 8.1%, where the prediction was
Egyptian < Saudi < MSA.

**The residual effect is smaller than the disagreement between metrics.** On chrF++ the aligned ordering
is MSA 6.3 < Vietnamese 7.0 < Korean 8.3; on COMET-KIWI it is Vietnamese 5.9 < Korean 6.8 < MSA 7.4. The
two metrics do not rank the languages the same way, and the whole spread is about 2 points. With three
to five languages and 300 to 500 test sentences each, no reliable word-order effect is detectable once
chunking is controlled.

The honest headline is therefore not "word order drives degradation" but **"chunk quality dominates the
apparent word-order effect; once it is controlled, what remains is within measurement noise."** That is
a methodological result rather than a typological one, and it comes with a concrete recommendation:
build simultaneous training data from word alignment, not from asking a model to segment.

### Final ordering, with every system processed identically

| Language | Word order | Degradation (best chunking) |
|---|---|---:|
| Vietnamese | SVO | **5.3%** |
| Saudi | VSO-leaning | 7.5% |
| MSA | VSO | 7.6% |
| Korean | SOV | 7.8% |
| Egyptian | SVO | 8.3% |

Vietnamese is clearly ahead. The other four sit inside 0.8 points of each other, which is smaller than
the gap between our two metrics on the same systems, so they are not distinguishable at this sample
size. Korean, which looked dramatically worst under the original chunking (23%), is now mid-pack.

**This does not rescue the hypothesis.** Vietnamese being best fits the prediction that SVO is easiest,
but Egyptian is also SVO and comes last, and Korean's SOV penalty has disappeared. A word-order account
cannot explain both. What separates Vietnamese is more mundane: it is the easiest pair overall
(chrF++ 49.3 against 21.6 for Korean) and it gained the most from the punctuation fix.

**Conclusion.** Across 20 trained systems, the study finds no coherent word-order effect on latency
degradation once chunk quality is controlled. What it does find is that the way training chunks are
built moves the result by up to 9 points, roughly five times the entire spread between languages.

**Caveats.** Vietnamese, MSA and Korean come from TED while Saudi and Egyptian come from Alexandria
conversation, so the cross-corpus comparison carries a domain confound. Korean's absolute chrF++ (17-22)
is depressed by its morphology and by free reference translations, which inspection of the predictions
confirmed: the outputs are fluent and accurate but share few n-grams with the reference. Degradation is
normalised within a language, so it is the fairer measure, but the absolute gap between languages is not
a quality ranking. At low latency several systems show AL at or below zero, meaning they start writing
almost immediately; whether that is genuine anticipation or a decoding artifact is not yet checked.
