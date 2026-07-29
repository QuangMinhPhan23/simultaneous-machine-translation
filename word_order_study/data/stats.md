# Step 1 - Data Sourcing Statistics

English source filtered to 15-30 whitespace words; deduped on English; robust target/source char-ratio misalignment filter.

**Lexical diversity.** `TTR` = target types / target tokens (whitespace-split, lowercased) over the whole split. Raw TTR falls as a corpus grows (tokens get reused), so it is *not* comparable across the differently-sized splits; `MATTR` (mean TTR over sliding windows of 100 target tokens, Covington & McFall 2010) removes that size dependence and is the column to compare across splits. **Both** still reflect segmentation/morphology, which differ by script (Vietnamese=syllables, Korean=eojeol, Arabic=words), so *cross-language* values measure tokenization + morphological richness as much as diversity - read them within a language, not as a cross-language ranking.

| Language | Order | Split | Pairs | Avg src words | Avg tgt words | Avg src chars | Avg tgt chars | TTR (tgt) | MATTR (tgt, W=100) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vietnamese | SVO | train | 2400 | 21.2 | 25.75 | 105.4 | 105.6 | 0.0563 | 0.7759 |
| vietnamese | SVO | dev | 300 | 20.92 | 25.51 | 104.8 | 104.7 | 0.1743 | 0.777 |
| vietnamese | SVO | test | 300 | 20.63 | 25.11 | 103.2 | 103.0 | 0.1741 | 0.7756 |
| msa | VSO | train | 2400 | 20.93 | 17.2 | 116.2 | 95.8 | 0.4035 | 0.8964 |
| msa | VSO | dev | 300 | 21.09 | 17.01 | 116.8 | 95.3 | 0.6213 | 0.9038 |
| msa | VSO | test | 300 | 20.74 | 17.15 | 116.5 | 95.9 | 0.6086 | 0.8963 |
| korean | SOV | train | 2400 | 21.08 | 14.44 | 115.7 | 60.1 | 0.4843 | 0.9458 |
| korean | SOV | dev | 300 | 21.1 | 14.64 | 116.3 | 61.6 | 0.6844 | 0.9469 |
| korean | SOV | test | 300 | 20.82 | 14.29 | 112.9 | 59.8 | 0.6865 | 0.9402 |
| saudi | VSO-leaning | train | 3000 | 20.93 | 14.81 | 118.5 | 83.8 | 0.351 | 0.9419 |
| saudi | VSO-leaning | dev | 500 | 21.14 | 15.35 | 119.5 | 85.9 | 0.5445 | 0.9313 |
| saudi | VSO-leaning | test | 500 | 21.01 | 15.06 | 118.1 | 84.2 | 0.5499 | 0.9375 |
| egyptian | SVO | train | 1925 | 20.74 | 15.58 | 115.4 | 87.5 | 0.3672 | 0.9228 |
| egyptian | SVO | dev | 500 | 20.71 | 15.17 | 115.7 | 85.0 | 0.5216 | 0.9288 |
| egyptian | SVO | test | 500 | 20.67 | 15.55 | 114.8 | 86.9 | 0.5215 | 0.922 |
