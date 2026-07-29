# Egyptian Arabic SimulMT - Results

Goal: make EAST do live English → **spoken Egyptian Arabic**. Tested on Alexandria (Egyptian split).
Scores: higher = better; latency = Average Lagging (AL), lower = more "live".

## The one lesson that keeps repeating

**A smaller model that already knows the dialect (Nile-Chat, 4B) beats a bigger one that doesn't
(EAST-8B, 8B) at almost everything** - we saw it ~4 separate times. What the base model already read
about the dialect matters more than its size or any training trick.

## Findings by phase

**1. Zero-shot (no training).** EAST-8B on Egyptian scored near zero (BLEU ~1.4) and answered in *formal*
MSA, not dialect. Not the prompt (a "be casual" instruction didn't help) and not Egyptian-specific (same
on Lebanese). Plain-translation check: EAST-8B **1.53** vs Nile-Chat **14.06** → the base model, not the
method, is the problem.

**2. Fine-tuning + replay.** Fine-tuning EAST-8B on ~12k Egyptian examples raised BLEU 1.35 → **8.27**,
but it *forgot German* (31 → 23). Mixing a little old data back in ("replay") recovered German (33) while
keeping Egyptian. Fine-tuning Nile-Chat instead beat EAST-8B at every latency.

**3. DPO (good-vs-bad training).** First attempt made things worse - the model hallucinated fluently.
Fix: an anchor term that keeps it faithful to the meaning. Result: formal-Arabic leakage **7.8% → 4.9%**
(the win is less MSA contamination, not a higher score).

**4. Chunking.** Feeding the whole sentence scored highest but had flat latency - it was secretly doing
offline translation, not live. The chunking that *varies its cuts per latency* is what teaches live
behaviour. Found and fixed **4 silent data bugs** (e.g. a chunker that never produced its 3 versions,
another mis-cutting Arabic); alignment quality **~87% → ~95%**. Nile-Chat beat EAST-8B in **24 of 25**
comparisons.

**5-6. Fairer evaluation.** No train/test leakage. Full 1,118-sentence test set (not just 103). On the
cross-paper ruler **chrF++, Nile-Chat 42.96 vs EAST-8B 37.86** - the gap is real, not a BLEU artifact.
A cheap CPT preview (raw Gemma vs Nile-Chat, same pipeline) showed extra Egyptian pre-training helps only
**~+1.5 BLEU / +0.5 COMET**, so we **decided not to spend ~65 GPU-hours** on the full pre-training run.

## Where this sits vs other work

On the Alexandria leaderboard, the best *open* models score ~28-30 chrF++ but are huge (27-111B); the
best ~9B model ~21; plain Gemma-4B 13.6. Their human reviewers rate even top models only **~3.3/5** on
"sounds like real dialect" - models get the meaning but stay formal. That is exactly this project's
target, so a *small* Egyptian-specialised model has room to stand out.

## Next

Skip expensive pre-training; instead **improve the fine-tuning data** (more Egyptian, more casual, less
formal) and re-test on the full set - likely closes the gap for little cost.
