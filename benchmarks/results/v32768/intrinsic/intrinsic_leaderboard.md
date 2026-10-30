- requested V = 32768; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `b8f3504b3e`; generated 2026-10-30T07:21:06Z
- arguments: `--train-bytes 10000000 --vocab-size 32768 --output-dir benchmarks/results/v32768/intrinsic`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 1499 identifiers mined from held-out code; throughput: median of 5 passes over 1.1 MB (206 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 32768 | 3.965 | 1.942 | 0.814 | 0.441 | 0.275 | 0.760 | 5.29 | 51.34 |
| `wp_bpe_base` | 32768 | 1.007 | 6.718 | 0.366 | 0.272 | 0.206 | 0.247 | 30.98 | 44.78 |
| `wp_bpe_ft` | 32768 | 1.021 | 6.627 | 0.364 | 0.269 | 0.206 | 0.247 | 29.45 | 43.11 |
| `wordpiece_pmi` | 32768 | 1.033 | 6.537 | 0.363 | 0.268 | 0.214 | 0.247 | 27.81 | 42.63 |
| `wp_bpe_scp` | 32768 | 2.087 | 2.897 | 0.514 | 0.231 | 0.269 | 0.497 | 13.80 | 34.23 |
| `morpheme_bpe` | 32768 | 2.743 | 1.967 | 0.596 | 0.189 | 0.411 | 0.754 | 8.90 | 29.96 |
| `wp_bpe_pig` | 32768 | 2.722 | 1.979 | 0.583 | 0.187 | 0.299 | 0.776 | 11.72 | 38.51 |
| `bpe_frequency` | 32768 | 2.791 | 1.916 | 0.602 | 0.186 | 0.277 | 0.770 | 11.15 | 40.02 |
| `length_penalized` | 32768 | 2.796 | 1.911 | 0.605 | 0.186 | 0.274 | 0.771 | 10.74 | 37.24 |
| `wp_bpe_ppig` | 32768 | 2.730 | 1.962 | 0.584 | 0.185 | 0.294 | 0.782 | 11.54 | 38.32 |
| `wp_bpe_npmi` | 32768 | 2.613 | 2.050 | 0.564 | 0.185 | 0.312 | 0.748 | 11.83 | 38.84 |
| `unigram_lm` | 32768 | 2.711 | 1.960 | 0.586 | 0.183 | 0.321 | 0.768 | 4.53 | 27.46 |
| `mdl_bits` | 32768 | 2.468 | 1.982 | 0.531 | 0.157 | 0.316 | 0.784 | 11.74 | 37.69 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.309 | 0.153 | 0.501 | 0.157 | 0.258 | 0.380 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wordpiece_pmi` | 0.212 | 0.193 | 0.344 | 0.126 | 0.196 | 0.268 |
| `wp_bpe_scp` | 0.314 | 0.194 | 0.426 | 0.169 | 0.241 | 0.346 |
| `morpheme_bpe` | 0.473 | 0.325 | 0.522 | 0.365 | 0.367 | 0.474 |
| `wp_bpe_pig` | 0.344 | 0.186 | 0.530 | 0.156 | 0.279 | 0.408 |
| `bpe_frequency` | 0.310 | 0.155 | 0.503 | 0.156 | 0.260 | 0.381 |
| `length_penalized` | 0.305 | 0.152 | 0.497 | 0.159 | 0.256 | 0.376 |
| `wp_bpe_ppig` | 0.342 | 0.177 | 0.532 | 0.158 | 0.260 | 0.407 |
| `wp_bpe_npmi` | 0.376 | 0.208 | 0.522 | 0.163 | 0.289 | 0.417 |
| `unigram_lm` | 0.398 | 0.231 | 0.514 | 0.220 | 0.243 | 0.421 |
| `mdl_bits` | 0.397 | 0.230 | 0.507 | 0.165 | 0.279 | 0.417 |
