# Step 2 - Tokenizer Fertility

## Target side (Llama-3-8B-Instruct tokenizer)

| Language | Script | n | chars/token | tokens/sentence | tokens/word | chars/sentence |
|---|---|---:|---:|---:|---:|---:|
| vietnamese | Latin + diacritics | 1000 | 3.661 | 28.7 | 1.117 | 105.1 |
| egyptian | Arabic | 1000 | 2.418 | 36.4 | 2.326 | 88.0 |
| saudi | Arabic | 1000 | 2.476 | 33.5 | 2.280 | 83.0 |
| msa | Arabic | 1000 | 2.545 | 37.6 | 2.188 | 95.7 |
| korean | Hangul | 1000 | 1.682 | 35.4 | 2.471 | 59.6 |

## English source side (read-side reference, same rows)

| Language | chars/token | tokens/sentence | tokens/word |
|---|---:|---:|---:|
| vietnamese | 4.456 | 23.5 | 1.114 |
| egyptian | 4.541 | 25.5 | 1.224 |
| saudi | 4.602 | 25.6 | 1.224 |
| msa | 4.573 | 25.5 | 1.215 |
| korean | 4.529 | 25.4 | 1.213 |

## Token-based AL inflation index (target tokens/sentence ÷ lowest)

Higher = the same 15-30-word English content is emitted as more target tokens, so a token-defined AL is inflated for that language. Report AL in **words/chars** too.

| Language | tokens/sentence | AL-inflation vs lowest |
|---|---:|---:|
| vietnamese | 28.7 | 1.00x |
| egyptian | 36.4 | 1.27x |
| saudi | 33.5 | 1.17x |
| msa | 37.6 | 1.31x |
| korean | 35.4 | 1.23x |
