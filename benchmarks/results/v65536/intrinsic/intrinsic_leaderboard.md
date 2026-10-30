- requested V = 65536; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `b8f3504b3e`; generated 2026-10-30T07:27:03Z
- arguments: `--train-bytes 10000000 --vocab-size 65536 --output-dir benchmarks/results/v65536/intrinsic`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 1499 identifiers mined from held-out code; throughput: median of 5 passes over 1.1 MB (206 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 65536 | 4.390 | 1.783 | 0.790 | 0.418 | 0.278 | 0.806 | 7.28 | 90.77 |
| `wp_bpe_base` | 65536 | 1.013 | 6.692 | 0.343 | 0.254 | 0.206 | 0.247 | 43.80 | 65.83 |
| `wp_bpe_ft` | 65536 | 1.053 | 6.388 | 0.338 | 0.248 | 0.214 | 0.247 | 38.69 | 61.77 |
| `wordpiece_pmi` | 65536 | 1.056 | 6.362 | 0.337 | 0.248 | 0.214 | 0.247 | 37.80 | 62.41 |
| `wp_bpe_scp` | 65536 | 2.513 | 2.238 | 0.515 | 0.189 | 0.309 | 0.666 | 17.62 | 59.34 |
| `wp_bpe_pig` | 65536 | 2.873 | 1.826 | 0.549 | 0.168 | 0.300 | 0.802 | 15.79 | 63.45 |
| `morpheme_bpe` | 65536 | 2.907 | 1.801 | 0.561 | 0.168 | 0.407 | 0.793 | 12.61 | 54.09 |
| `wp_bpe_ppig` | 65536 | 2.888 | 1.808 | 0.550 | 0.167 | 0.298 | 0.811 | 15.52 | 68.04 |
| `bpe_frequency` | 65536 | 2.944 | 1.765 | 0.564 | 0.166 | 0.280 | 0.807 | 13.33 | 57.37 |
| `length_penalized` | 65536 | 2.951 | 1.759 | 0.566 | 0.166 | 0.279 | 0.808 | 15.52 | 68.68 |
| `wp_bpe_npmi` | 65536 | 2.756 | 1.892 | 0.531 | 0.166 | 0.318 | 0.788 | 16.79 | 65.50 |
| `unigram_lm` | 65536 | 2.828 | 1.836 | 0.549 | 0.166 | 0.338 | 0.797 | 7.09 | 57.45 |
| `mdl_bits` | 40496 | 2.509 | 1.928 | 0.521 | 0.152 | 0.318 | 0.796 | 17.61 | 62.33 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.314 | 0.162 | 0.509 | 0.152 | 0.253 | 0.385 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.193 | 0.344 | 0.126 | 0.196 | 0.268 |
| `wordpiece_pmi` | 0.212 | 0.193 | 0.344 | 0.126 | 0.196 | 0.268 |
| `wp_bpe_scp` | 0.367 | 0.226 | 0.496 | 0.173 | 0.284 | 0.403 |
| `wp_bpe_pig` | 0.350 | 0.190 | 0.537 | 0.155 | 0.265 | 0.413 |
| `morpheme_bpe` | 0.465 | 0.306 | 0.534 | 0.370 | 0.358 | 0.476 |
| `wp_bpe_ppig` | 0.348 | 0.184 | 0.540 | 0.160 | 0.258 | 0.413 |
| `bpe_frequency` | 0.315 | 0.164 | 0.511 | 0.152 | 0.255 | 0.387 |
| `length_penalized` | 0.311 | 0.163 | 0.507 | 0.158 | 0.253 | 0.384 |
| `wp_bpe_npmi` | 0.388 | 0.219 | 0.532 | 0.164 | 0.286 | 0.426 |
| `unigram_lm` | 0.411 | 0.246 | 0.539 | 0.241 | 0.253 | 0.441 |
| `mdl_bits` | 0.400 | 0.233 | 0.512 | 0.163 | 0.284 | 0.421 |
