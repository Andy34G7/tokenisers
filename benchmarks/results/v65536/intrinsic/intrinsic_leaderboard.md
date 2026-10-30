- requested V = 65536; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T13:35:36Z
- arguments: `--skip-memory-check --vocab-size 65536 --output-dir benchmarks/results/v65536/intrinsic`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 2000 identifiers mined from held-out code; throughput: median of 5 passes over 3.1 MB (666 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 65536 | 4.537 | 1.718 | 0.832 | 0.405 | 0.283 | 0.847 | 5.26 | 58.37 |
| `wp_bpe_base` | 65536 | 1.011 | 6.722 | 0.349 | 0.258 | 0.206 | 0.245 | 27.40 | 40.99 |
| `wp_bpe_ft` | 65536 | 1.022 | 6.632 | 0.348 | 0.257 | 0.206 | 0.245 | 30.80 | 43.40 |
| `wordpiece_pmi` | 65536 | 1.025 | 6.612 | 0.348 | 0.256 | 0.206 | 0.245 | 29.38 | 43.09 |
| `wp_bpe_scp` | 65536 | 2.254 | 2.593 | 0.501 | 0.204 | 0.278 | 0.605 | 15.05 | 43.77 |
| `wp_bpe_npmi` | 65536 | 2.888 | 1.789 | 0.556 | 0.164 | 0.320 | 0.840 | 11.78 | 45.31 |
| `morpheme_bpe` | 65536 | 2.957 | 1.742 | 0.580 | 0.164 | 0.445 | 0.839 | 11.96 | 47.62 |
| `wp_bpe_pig` | 65536 | 2.964 | 1.732 | 0.570 | 0.163 | 0.303 | 0.850 | 10.96 | 42.68 |
| `unigram_lm` | 65536 | 2.898 | 1.770 | 0.570 | 0.162 | 0.333 | 0.849 | 6.65 | 48.68 |
| `wp_bpe_ppig` | 65536 | 2.976 | 1.720 | 0.572 | 0.162 | 0.303 | 0.857 | 11.58 | 47.66 |
| `length_penalized` | 65536 | 3.007 | 1.696 | 0.587 | 0.161 | 0.282 | 0.848 | 12.28 | 50.72 |
| `bpe_frequency` | 65536 | 3.008 | 1.696 | 0.585 | 0.161 | 0.285 | 0.849 | 11.51 | 46.70 |
| `mdl_bits` | 65536 | 2.687 | 1.746 | 0.519 | 0.139 | 0.322 | 0.854 | 11.67 | 44.10 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.319 | 0.169 | 0.516 | 0.167 | 0.247 | 0.391 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wordpiece_pmi` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_scp` | 0.324 | 0.200 | 0.436 | 0.178 | 0.252 | 0.356 |
| `wp_bpe_npmi` | 0.385 | 0.219 | 0.540 | 0.176 | 0.279 | 0.429 |
| `morpheme_bpe` | 0.488 | 0.336 | 0.557 | 0.448 | 0.397 | 0.504 |
| `wp_bpe_pig` | 0.350 | 0.189 | 0.546 | 0.165 | 0.266 | 0.418 |
| `unigram_lm` | 0.420 | 0.251 | 0.537 | 0.228 | 0.226 | 0.440 |
| `wp_bpe_ppig` | 0.350 | 0.189 | 0.545 | 0.169 | 0.264 | 0.417 |
| `length_penalized` | 0.317 | 0.166 | 0.507 | 0.170 | 0.251 | 0.386 |
| `bpe_frequency` | 0.320 | 0.171 | 0.518 | 0.167 | 0.248 | 0.392 |
| `mdl_bits` | 0.402 | 0.236 | 0.523 | 0.169 | 0.281 | 0.427 |
