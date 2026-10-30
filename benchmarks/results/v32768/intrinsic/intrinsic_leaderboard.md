- requested V = 32768; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T13:12:14Z
- arguments: `--skip-memory-check --vocab-size 32768 --output-dir benchmarks/results/v32768/intrinsic`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 2000 identifiers mined from held-out code; throughput: median of 5 passes over 3.1 MB (666 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 32768 | 4.062 | 1.886 | 0.848 | 0.427 | 0.283 | 0.828 | 7.41 | 84.08 |
| `wp_bpe_base` | 32768 | 1.011 | 6.723 | 0.372 | 0.276 | 0.206 | 0.245 | 42.65 | 63.01 |
| `wp_bpe_ft` | 32768 | 1.020 | 6.652 | 0.371 | 0.274 | 0.206 | 0.245 | 38.84 | 57.86 |
| `wordpiece_pmi` | 32768 | 1.020 | 6.650 | 0.371 | 0.274 | 0.206 | 0.245 | 43.03 | 64.32 |
| `wp_bpe_scp` | 32768 | 1.884 | 3.296 | 0.497 | 0.245 | 0.260 | 0.436 | 17.89 | 45.19 |
| `morpheme_bpe` | 32768 | 2.748 | 1.950 | 0.613 | 0.186 | 0.438 | 0.805 | 14.24 | 56.51 |
| `wp_bpe_npmi` | 32768 | 2.720 | 1.958 | 0.587 | 0.184 | 0.316 | 0.816 | 16.04 | 62.68 |
| `wp_bpe_pig` | 32768 | 2.791 | 1.900 | 0.602 | 0.183 | 0.302 | 0.825 | 15.06 | 62.67 |
| `unigram_lm` | 32768 | 2.753 | 1.921 | 0.606 | 0.182 | 0.327 | 0.823 | 10.41 | 67.15 |
| `wp_bpe_ppig` | 32768 | 2.804 | 1.885 | 0.605 | 0.182 | 0.299 | 0.835 | 15.67 | 63.82 |
| `length_penalized` | 32768 | 2.836 | 1.858 | 0.623 | 0.181 | 0.278 | 0.831 | 16.95 | 69.13 |
| `bpe_frequency` | 32768 | 2.838 | 1.857 | 0.620 | 0.181 | 0.282 | 0.833 | 14.13 | 60.01 |
| `mdl_bits` | 32768 | 2.536 | 1.926 | 0.550 | 0.158 | 0.321 | 0.825 | 17.28 | 63.21 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.313 | 0.158 | 0.507 | 0.170 | 0.266 | 0.386 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wordpiece_pmi` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_scp` | 0.288 | 0.193 | 0.391 | 0.175 | 0.255 | 0.323 |
| `morpheme_bpe` | 0.492 | 0.346 | 0.539 | 0.433 | 0.379 | 0.495 |
| `wp_bpe_npmi` | 0.379 | 0.215 | 0.525 | 0.174 | 0.289 | 0.420 |
| `wp_bpe_pig` | 0.350 | 0.190 | 0.535 | 0.165 | 0.271 | 0.413 |
| `unigram_lm` | 0.408 | 0.238 | 0.525 | 0.213 | 0.250 | 0.430 |
| `wp_bpe_ppig` | 0.344 | 0.180 | 0.537 | 0.168 | 0.265 | 0.411 |
| `length_penalized` | 0.308 | 0.155 | 0.501 | 0.169 | 0.255 | 0.380 |
| `bpe_frequency` | 0.314 | 0.158 | 0.510 | 0.168 | 0.259 | 0.387 |
| `mdl_bits` | 0.402 | 0.232 | 0.507 | 0.167 | 0.297 | 0.420 |
