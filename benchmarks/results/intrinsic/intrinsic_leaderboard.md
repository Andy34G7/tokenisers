- requested V = 16384; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `df4c7d8edc`; generated 2026-10-30T05:15:19Z
- arguments: `--train-bytes 10000000`
- morph F1: 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)); identifier F1: 1499 identifiers mined from held-out code; throughput: median of 5 passes over 1.1 MB (206 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 16384 | 3.559 | 2.137 | 0.836 | 0.464 | 0.369 | 0.721 | 6.94 | 84.57 |
| `wp_bpe_base` | 16384 | 1.007 | 6.720 | 0.392 | 0.291 | 0.252 | 0.247 | 44.22 | 61.74 |
| `wp_bpe_ft` | 16384 | 1.016 | 6.664 | 0.391 | 0.290 | 0.252 | 0.247 | 43.35 | 62.31 |
| `wordpiece_pmi` | 16384 | 1.017 | 6.662 | 0.391 | 0.289 | 0.252 | 0.247 | 45.21 | 62.20 |
| `wp_bpe_scp` | 16384 | 1.651 | 3.921 | 0.494 | 0.285 | 0.309 | 0.357 | 28.54 | 53.71 |
| `morpheme_bpe` | 16384 | 2.534 | 2.212 | 0.632 | 0.216 | 0.464 | 0.683 | 17.32 | 54.63 |
| `wp_bpe_pig` | 16384 | 2.545 | 2.178 | 0.617 | 0.211 | 0.395 | 0.717 | 21.39 | 66.88 |
| `length_penalized` | 16384 | 2.620 | 2.106 | 0.643 | 0.210 | 0.365 | 0.722 | 19.45 | 64.72 |
| `bpe_frequency` | 16384 | 2.620 | 2.106 | 0.640 | 0.210 | 0.371 | 0.730 | 18.54 | 62.76 |
| `unigram_lm` | 16384 | 2.574 | 2.131 | 0.628 | 0.207 | 0.408 | 0.715 | 10.70 | 56.74 |
| `wp_bpe_npmi` | 16384 | 2.442 | 2.253 | 0.596 | 0.207 | 0.403 | 0.706 | 22.31 | 71.97 |
| `wp_bpe_ppig` | 16384 | 2.544 | 2.156 | 0.616 | 0.206 | 0.398 | 0.729 | 21.80 | 69.89 |
| `mdl_bits` | 16384 | 2.316 | 2.199 | 0.565 | 0.180 | 0.407 | 0.720 | 22.04 | 66.61 |
