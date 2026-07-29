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

## Result 3 - training (Step 4, in progress)

10 systems (5 languages × 2 chunking methods), LoRA Stage-II on the EAST-8B base, adapters pushed to
Hugging Face (`Henry236/east8b-<lang>-lora-<method>`). First adapter (korean/generic) trained and
verified in ~27 min on one A100; the rest are running.

## The real test (Step 5, pending)

Quality (chrF++ and spBLEU) vs latency (AL in words), one curve per system. Degradation =
(chrF_high - chrF_low) / chrF_high. The hypothesis holds if degradation follows the predicted order, and
is strongest for the **within-Arabic** comparison (Egyptian SVO vs Saudi vs MSA), where script and
morphology are held constant and only word order changes.
