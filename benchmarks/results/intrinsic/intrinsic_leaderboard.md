- requested V = 16384; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `b8f3504b3e`; generated 2026-10-30T07:18:20Z
- arguments: `--train-bytes 10000000`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 1499 identifiers mined from held-out code; throughput: median of 5 passes over 1.1 MB (206 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 16384 | 3.559 | 2.137 | 0.836 | 0.464 | 0.273 | 0.721 | 6.16 | 52.41 |
| `wp_bpe_base` | 16384 | 1.007 | 6.720 | 0.392 | 0.291 | 0.206 | 0.247 | 31.21 | 44.80 |
| `wp_bpe_ft` | 16384 | 1.016 | 6.664 | 0.391 | 0.290 | 0.206 | 0.247 | 30.26 | 43.87 |
| `wordpiece_pmi` | 16384 | 1.017 | 6.662 | 0.391 | 0.289 | 0.206 | 0.247 | 30.14 | 45.36 |
| `wp_bpe_scp` | 16384 | 1.651 | 3.921 | 0.494 | 0.285 | 0.249 | 0.357 | 16.66 | 33.24 |
| `morpheme_bpe` | 16384 | 2.534 | 2.212 | 0.632 | 0.216 | 0.404 | 0.683 | 12.40 | 40.38 |
| `wp_bpe_pig` | 16384 | 2.545 | 2.178 | 0.617 | 0.211 | 0.295 | 0.717 | 12.71 | 41.22 |
| `length_penalized` | 16384 | 2.620 | 2.106 | 0.643 | 0.210 | 0.269 | 0.722 | 12.46 | 40.39 |
| `bpe_frequency` | 16384 | 2.620 | 2.106 | 0.640 | 0.210 | 0.273 | 0.730 | 12.35 | 40.73 |
| `unigram_lm` | 16384 | 2.574 | 2.131 | 0.628 | 0.207 | 0.315 | 0.715 | 7.95 | 40.57 |
| `wp_bpe_npmi` | 16384 | 2.442 | 2.253 | 0.596 | 0.207 | 0.307 | 0.706 | 13.35 | 39.98 |
| `wp_bpe_ppig` | 16384 | 2.544 | 2.156 | 0.616 | 0.206 | 0.292 | 0.729 | 12.93 | 40.46 |
| `mdl_bits` | 16384 | 2.316 | 2.199 | 0.565 | 0.180 | 0.315 | 0.720 | 13.18 | 40.16 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.303 | 0.154 | 0.481 | 0.163 | 0.267 | 0.369 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wordpiece_pmi` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_scp` | 0.262 | 0.180 | 0.382 | 0.179 | 0.242 | 0.309 |
| `morpheme_bpe` | 0.475 | 0.326 | 0.504 | 0.357 | 0.359 | 0.464 |
| `wp_bpe_pig` | 0.334 | 0.179 | 0.508 | 0.159 | 0.294 | 0.395 |
| `length_penalized` | 0.296 | 0.147 | 0.477 | 0.162 | 0.264 | 0.365 |
| `bpe_frequency` | 0.302 | 0.154 | 0.486 | 0.160 | 0.265 | 0.371 |
| `unigram_lm` | 0.389 | 0.222 | 0.490 | 0.209 | 0.266 | 0.408 |
| `wp_bpe_npmi` | 0.364 | 0.205 | 0.498 | 0.170 | 0.298 | 0.403 |
| `wp_bpe_ppig` | 0.334 | 0.173 | 0.516 | 0.160 | 0.280 | 0.398 |
| `mdl_bits` | 0.386 | 0.227 | 0.490 | 0.170 | 0.301 | 0.407 |
