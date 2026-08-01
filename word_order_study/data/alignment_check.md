# Chunk alignment check

`matched` = mean similarity of English chunk i to target chunk i.
`shuffled` = the same chunks paired with someone else's partner (control).
`gap` = matched - shuffled. A large gap means chunk i really does correspond to chunk i.
A gap near zero means the chunks are misaligned, even if they rebuild the sentence.

| Language | Method | Latency | Chunks from | matched | shuffled | gap | n |
|---|---|---|---|---:|---:|---:|---:|
| vietnamese | generic | low | llm | 0.649 | 0.273 | +0.376 | 600 |
| vietnamese | generic | low | fallback | 0.594 | 0.470 | +0.125 | 600 |
| vietnamese | generic | medium | llm | 0.767 | 0.151 | +0.616 | 600 |
| vietnamese | generic | medium | fallback | 0.606 | 0.304 | +0.302 | 600 |
| vietnamese | generic | high | llm | 0.845 | 0.115 | +0.730 | 600 |
| vietnamese | generic | high | fallback | 0.682 | 0.189 | +0.493 | 600 |
| vietnamese | specific | low | llm | 0.649 | 0.274 | +0.375 | 600 |
| vietnamese | specific | low | fallback | 0.605 | 0.489 | +0.116 | 600 |
| vietnamese | specific | medium | llm | 0.756 | 0.146 | +0.609 | 600 |
| vietnamese | specific | medium | fallback | 0.613 | 0.303 | +0.310 | 600 |
| vietnamese | specific | high | llm | 0.843 | 0.116 | +0.726 | 600 |
| vietnamese | specific | high | fallback | 0.688 | 0.203 | +0.485 | 600 |
| msa | generic | low | llm | 0.672 | 0.256 | +0.416 | 600 |
| msa | generic | low | fallback | 0.521 | 0.369 | +0.152 | 600 |
| msa | generic | medium | llm | 0.724 | 0.145 | +0.578 | 600 |
| msa | generic | medium | fallback | 0.547 | 0.229 | +0.318 | 600 |
| msa | generic | high | llm | 0.828 | 0.119 | +0.709 | 600 |
| msa | generic | high | fallback | 0.625 | 0.283 | +0.343 | 433 |
| msa | specific | low | llm | 0.649 | 0.286 | +0.363 | 600 |
| msa | specific | low | fallback | 0.542 | 0.372 | +0.170 | 600 |
| msa | specific | medium | llm | 0.734 | 0.144 | +0.590 | 600 |
| msa | specific | medium | fallback | 0.574 | 0.240 | +0.334 | 600 |
| msa | specific | high | llm | 0.831 | 0.108 | +0.723 | 600 |
| msa | specific | high | fallback | 0.620 | 0.276 | +0.344 | 504 |
| korean | generic | low | llm | 0.460 | 0.310 | +0.150 | 600 |
| korean | generic | low | fallback | 0.462 | 0.402 | +0.060 | 600 |
| korean | generic | medium | llm | 0.555 | 0.170 | +0.385 | 600 |
| korean | generic | medium | fallback | 0.417 | 0.282 | +0.135 | 600 |
| korean | generic | high | llm | 0.801 | 0.124 | +0.676 | 600 |
| korean | generic | high | fallback | 0.499 | 0.195 | +0.305 | 600 |
| korean | specific | low | llm | 0.465 | 0.314 | +0.151 | 600 |
| korean | specific | low | fallback | 0.463 | 0.383 | +0.080 | 600 |
| korean | specific | medium | llm | 0.538 | 0.169 | +0.369 | 600 |
| korean | specific | medium | fallback | 0.415 | 0.263 | +0.152 | 600 |
| korean | specific | high | llm | 0.799 | 0.126 | +0.673 | 600 |
| korean | specific | high | fallback | 0.496 | 0.190 | +0.306 | 600 |
| saudi | generic | low | llm | 0.564 | 0.249 | +0.316 | 600 |
| saudi | generic | low | fallback | 0.526 | 0.365 | +0.162 | 600 |
| saudi | generic | medium | llm | 0.627 | 0.148 | +0.479 | 600 |
| saudi | generic | medium | fallback | 0.488 | 0.237 | +0.251 | 600 |
| saudi | generic | high | llm | 0.694 | 0.136 | +0.558 | 600 |
| saudi | generic | high | fallback | 0.525 | 0.243 | +0.283 | 214 |
| saudi | specific | low | llm | 0.534 | 0.267 | +0.267 | 600 |
| saudi | specific | low | fallback | 0.524 | 0.374 | +0.151 | 600 |
| saudi | specific | medium | llm | 0.606 | 0.145 | +0.461 | 600 |
| saudi | specific | medium | fallback | 0.498 | 0.226 | +0.272 | 600 |
| saudi | specific | high | llm | 0.687 | 0.144 | +0.543 | 600 |
| saudi | specific | high | fallback | 0.557 | 0.244 | +0.313 | 185 |
| egyptian | generic | low | llm | 0.533 | 0.274 | +0.259 | 600 |
| egyptian | generic | low | fallback | 0.493 | 0.367 | +0.126 | 600 |
| egyptian | generic | medium | llm | 0.570 | 0.151 | +0.420 | 600 |
| egyptian | generic | medium | fallback | 0.463 | 0.306 | +0.157 | 501 |
| egyptian | generic | high | llm | 0.618 | 0.151 | +0.467 | 600 |
| egyptian | generic | high | fallback | 0.509 | 0.236 | +0.273 | 156 |
| egyptian | specific | low | llm | 0.524 | 0.302 | +0.222 | 600 |
| egyptian | specific | low | fallback | 0.481 | 0.368 | +0.114 | 600 |
| egyptian | specific | medium | llm | 0.566 | 0.153 | +0.414 | 600 |
| egyptian | specific | medium | fallback | 0.471 | 0.310 | +0.161 | 480 |
| egyptian | specific | high | llm | 0.604 | 0.135 | +0.470 | 600 |
| egyptian | specific | high | fallback | 0.504 | 0.235 | +0.269 | 140 |
