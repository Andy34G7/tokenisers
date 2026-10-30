- requested V = 16384; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T13:09:07Z
- arguments: `--skip-memory-check`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 2000 identifiers mined from held-out code; throughput: median of 5 passes over 3.1 MB (666 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 16384 | 3.600 | 2.101 | 0.860 | 0.453 | 0.277 | 0.780 | 8.14 | 81.10 |
| `wp_bpe_base` | 16384 | 1.007 | 6.737 | 0.399 | 0.296 | 0.206 | 0.245 | 39.21 | 57.49 |
| `wp_bpe_ft` | 16384 | 1.017 | 6.671 | 0.398 | 0.294 | 0.206 | 0.245 | 38.00 | 57.48 |
| `wordpiece_pmi` | 16384 | 1.017 | 6.670 | 0.398 | 0.294 | 0.206 | 0.245 | 43.89 | 67.22 |
| `wp_bpe_scp` | 16384 | 1.641 | 3.927 | 0.497 | 0.284 | 0.248 | 0.349 | 23.31 | 51.23 |
| `morpheme_bpe` | 16384 | 2.529 | 2.203 | 0.645 | 0.214 | 0.420 | 0.737 | 17.52 | 63.98 |
| `wp_bpe_npmi` | 16384 | 2.531 | 2.175 | 0.619 | 0.208 | 0.306 | 0.774 | 18.04 | 65.42 |
| `wp_bpe_pig` | 16384 | 2.600 | 2.112 | 0.635 | 0.208 | 0.295 | 0.789 | 18.84 | 69.27 |
| `wp_bpe_ppig` | 16384 | 2.609 | 2.098 | 0.638 | 0.207 | 0.294 | 0.793 | 18.55 | 70.89 |
| `length_penalized` | 16384 | 2.641 | 2.069 | 0.659 | 0.207 | 0.273 | 0.780 | 15.06 | 59.51 |
| `bpe_frequency` | 16384 | 2.643 | 2.066 | 0.656 | 0.206 | 0.279 | 0.789 | 17.64 | 68.85 |
| `unigram_lm` | 16384 | 2.588 | 2.108 | 0.643 | 0.206 | 0.322 | 0.773 | 12.23 | 69.62 |
| `mdl_bits` | 16384 | 2.371 | 2.147 | 0.583 | 0.180 | 0.311 | 0.782 | 16.42 | 57.87 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.304 | 0.152 | 0.492 | 0.171 | 0.267 | 0.375 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wordpiece_pmi` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_scp` | 0.260 | 0.179 | 0.378 | 0.185 | 0.237 | 0.306 |
| `morpheme_bpe` | 0.483 | 0.334 | 0.514 | 0.404 | 0.363 | 0.476 |
| `wp_bpe_npmi` | 0.361 | 0.201 | 0.505 | 0.177 | 0.285 | 0.405 |
| `wp_bpe_pig` | 0.340 | 0.181 | 0.517 | 0.168 | 0.267 | 0.400 |
| `wp_bpe_ppig` | 0.340 | 0.178 | 0.514 | 0.171 | 0.267 | 0.399 |
| `length_penalized` | 0.300 | 0.147 | 0.482 | 0.171 | 0.265 | 0.369 |
| `bpe_frequency` | 0.306 | 0.153 | 0.493 | 0.172 | 0.270 | 0.377 |
| `unigram_lm` | 0.400 | 0.235 | 0.508 | 0.200 | 0.267 | 0.420 |
| `mdl_bits` | 0.391 | 0.224 | 0.490 | 0.169 | 0.283 | 0.407 |
