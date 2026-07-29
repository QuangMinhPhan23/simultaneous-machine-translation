# Word-order divergence in simultaneous MT

**Question.** In simultaneous MT, does the word-order gap between the source and the target drive
quality loss at low latency? English is SVO (Subject-Verb-Object). If the target language puts words in
a very different order, the interpreter has to wait longer before it can start writing, which should
hurt more at low latency. I test this across five languages, holding the model and method fixed.

The novel part is the **within-Arabic** comparison: Egyptian (SVO), Saudi (VSO-leaning), and MSA (VSO)
share script, morphology, and family, so word order is almost the only thing that changes between them.

| Language | Word order | Script | Source |
|---|---|---|---|
| Vietnamese | SVO | Latin + diacritics | IWSLT15 En-Vi (TED) |
| Egyptian Arabic | SVO (dialect) | Arabic | Alexandria EG |
| Saudi Arabic | VSO-leaning (dialect) | Arabic | Alexandria SA |
| MSA | VSO | Arabic | TED2020 En-Ar |
| Korean | SOV | Hangul | TED (multitarget) |

**Prediction** (least to most degradation from high to low latency): Vietnamese < Egyptian < Saudi <
MSA < Korean.

Each language is run with **two chunking methods** (see Step 3), so 5 languages x 2 methods = 10 systems.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Data | `source_step1_data.py` | `data/<lang>/{train,dev,test}.{en,xx}` + `data/stats.md` |
| 2. Tokenizer | `measure_tokenizer_step2.py` | `data/tokenizer_fertility.md` |
| 3a. Chunking | `generate_chunks.py` | `data/<lang>/chunks-<method>.json` |
| 3b. SFT build | `build_simt_sft.py` | `data/<lang>/simt-<method>.train.json` |
| 4. Training | `submit_train.pbs` | LoRA adapter pushed to HF |

Helpers: `chunk_prompts.py` (the two prompts) and `chunk_validate.py` (checks a chunking rebuilds the
sentence). Steps 1 and 2 run locally; Steps 3 and 4 need a GPU and run on Katana via the two `.pbs`
scripts.

**The two chunking methods.** Both split each sentence into aligned read/write chunks at three latency
levels using a local Llama-3-8B-Instruct model. *Generic* uses one language-agnostic prompt; *specific*
adds a short word-order hint per language (e.g. "Korean is SOV, the verb comes last, so keep noun
phrases together"). Both share the exact same output format, so any difference comes from the hint, not
the format. When the model's chunking does not rebuild the sentence, we fall back to a simple word-count
split, and `generate_chunks.py` reports how often that happens.

## Status

- **Step 1-2 done** (local). Filters: English side 15-30 words, dedup on English, drop badly-aligned
  pairs. Sizes: 2400/300/300 (Vietnamese, MSA, Korean), up to 3000/500/500 (Saudi, Egyptian).
- **Step 3 done** on Katana (all 10 chunk sets generated).
- **Step 4 (training) running** on Katana.

Tokenizer result (Step 2, first 1000 train sentences), which matters because it is a confound for
token-based latency:

| Language | chars/token | tokens/sentence | AL-inflation vs lowest |
|---|---:|---:|---:|
| Vietnamese | 3.66 | 28.7 | 1.00x |
| Saudi | 2.48 | 33.5 | 1.17x |
| Korean | 1.68 | 35.4 | 1.23x |
| Egyptian | 2.42 | 36.4 | 1.27x |
| MSA | 2.55 | 37.6 | 1.31x |

Vietnamese produces ~25% fewer target tokens for the same English, so a token-based Average Lagging
would understate its latency. **Decision:** report latency in words (and characters), not only tokens.

**Honest caveat on the chunker fallback rate.** After fixing a punctuation-only validation bug, the
fallback rate stopped tracking word order (it mostly reflects how faithfully Llama copies each script),
so it is weak evidence for the hypothesis. The real test is the Step 5 quality-vs-latency curve, not the
fallback rate.
