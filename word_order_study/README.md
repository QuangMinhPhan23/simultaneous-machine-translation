# Word-order divergence in simultaneous MT

**Question.** In simultaneous MT, does the word-order gap between source and target drive quality loss
at low latency? English is SVO. If the target puts words in a very different order, the model has to
wait longer before it can start writing, which should hurt more at low latency. Tested across five
languages with the model and training recipe held fixed.

The novel part is the **within-Arabic** comparison: Egyptian (SVO), Saudi (VSO-leaning) and MSA (VSO)
share script, morphology and family, so word order is almost the only thing that changes.

| Language | Word order | Script | Source |
|---|---|---|---|
| Vietnamese | SVO | Latin + diacritics | IWSLT15 En-Vi (TED) |
| Egyptian Arabic | SVO (dialect) | Arabic | Alexandria EG |
| Saudi Arabic | VSO-leaning (dialect) | Arabic | Alexandria SA |
| MSA | VSO | Arabic | TED2020 En-Ar |
| Korean | SOV | Hangul | TED (multitarget) |

**Findings are in [RESULTS.md](RESULTS.md).**

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Data | `source_step1_data.py` | `data/<lang>/{train,dev,test}.{en,xx}` + `data/stats.md` |
| 2. Tokenizer | `measure_tokenizer_step2.py` | `data/tokenizer_fertility.md` |
| 3a. Chunking | `generate_chunks.py` (A, B) or `align_chunks.py` (C, D) | `data/<lang>/chunks-<method>.json` |
| 3b. SFT build | `build_simt_sft.py` | `data/<lang>/simt-<method>.train.json` |
| 4. Training | `submit_train.pbs` | LoRA adapter pushed to Hugging Face |
| 5. Test data | `prepare_test_data.py` | `data/<lang>/test.eval.json` |
| 6. Evaluation | `submit_eval.pbs` | `results/wordorder/<lang>-<method>/<latency>/` |

Steps 1, 2, 3b and 5 run locally. Steps 3a, 4 and 6 need a GPU and run on Katana through the `.pbs`
scripts. Helpers: `chunk_prompts.py` (the prompts), `chunk_validate.py` (checks a chunking rebuilds the
sentence), `segment.py` (Vietnamese word segmentation), `syntax_boundaries.py` (spaCy boundaries).

## The four chunking methods

Training data is a sentence pair cut into aligned read/write steps: step i of the English is glued to
step i of the target. If the steps do not correspond, the model is taught to write things it has not
read yet, so how the cuts are made decides what the model learns.

- **A, generic** - ask Llama-3-8B-Instruct to cut both sides into the same number of steps, at three
  granularities in one answer. One prompt for every language.
- **B, specific** - the same prompt plus a word-order hint, for example "Korean is SOV, the verb comes
  at the end". Everything else is identical, so a difference comes from the hint alone.
- **C, aligned** - no segmenting model. Align English words to target words with multilingual BERT,
  then cut where the alignment allows: a target word may only enter a step once every English word it
  corresponds to has been read. The target is only cut, never re-translated, so the human translation
  survives exactly.
- **D, syntactic** - Method C, but the English is cut at syntactic joints found with spaCy (end of a
  noun phrase, before a preposition or subordinating conjunction, after punctuation) instead of every
  k words, capped at 7 words per step.

A and B fall back to a word-count split when the model's answer cannot rebuild the sentence.
`build_simt_sft.py --merge_punct_only` folds away steps whose target side is only punctuation.

## Running a job

Submit from the repo root with the walltime on the command line. Leave the GPU unpinned; a job that
lands on the Blackwell node crashes in about 4 minutes and can just be resubmitted.

```bash
QS="-l walltime=2:00:00 -l select=1:ncpus=8:mem=48gb:ngpus=1"
qsub $QS -v LANGUAGE=korean,METHOD=aligned word_order_study/submit_train.pbs
qsub $QS -v LANGUAGE=korean,METHOD=aligned,LATENCY=low word_order_study/submit_eval.pbs
```

## Checking the data

- `check_word_splits.py` - how often a chunk boundary cuts a word in half, and how many steps write
  only punctuation.
- `check_alignment.py` - whether step i of the English really corresponds to step i of the target,
  against a shuffled control.
- `rescore_kiwi.py` - re-score finished predictions with COMET-KIWI, which does not use the reference.
- `chunk_examples_best.html` - worked examples for three languages at three latencies.
