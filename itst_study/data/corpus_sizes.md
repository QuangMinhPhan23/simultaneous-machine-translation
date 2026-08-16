# ITST corpus sizes

Built by `source_itst_data.py`. English length band 3-80 words, deduped on English, target/source character ratio kept within 1/3x to 3x of the language median. Dev and test are copied unchanged from `word_order_study/data/` and their English sentences are excluded from train.

| Dataset | Order | Raw pairs | In band | Deduped | Ratio-clean | Pool | Train | Dev | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vietnamese | SVO | 133166 | 132152 | 130728 | 130677 | 130325 | 130000 | 300 | 300 |
| msa | VSO | 363344 | 354866 | 351948 | 351252 | 350652 | 130000 | 300 | 300 |
| korean | SOV | 166215 | 163938 | 163119 | 162974 | 162374 | 130000 | 300 | 300 |
| saudi | VSO-leaning | 10693 | 10681 | 10657 | 10650 | 9650 | 9650 | 500 | 500 |
| saudi-matched | VSO-leaning | 10693 | 10681 | 10657 | 10650 | 9650 | 4323 | 500 | 500 |
| egyptian | SVO | 5339 | 5330 | 5325 | 5323 | 4323 | 4323 | 500 | 500 |
