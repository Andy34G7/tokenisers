- requested V = 16384; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T12:59:35Z
- arguments: (defaults)
- macro = unweighted mean over the 17 languages with >= 3 held-out documents and >= 10,000 bytes (en, de, fr, es, ru, zh, ja, ar, hi, hu, fi, python, javascript, java, cpp, go, rust); micro = totals summed over all 17 languages (token-weighted); unique tokens = union over languages

| Tokenizer | \|V\| | Macro bytes/token (↑) | Micro bytes/token (↑) | Macro fertility (↓) | Macro continued-token % (↓) | Unique tokens used |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 16384 | 3.432 | 3.599 | 2.606 | 53.03 | 15867 |
| `bpe_frequency` | 16384 | 2.787 | 2.643 | 3.237 | 52.27 | 15735 |
| `length_penalized` | 16384 | 2.784 | 2.640 | 3.239 | 52.15 | 15808 |
| `wp_bpe_ppig` | 16384 | 2.759 | 2.608 | 3.300 | 52.56 | 15319 |
| `wp_bpe_pig` | 16384 | 2.752 | 2.599 | 3.311 | 52.98 | 15151 |
| `unigram_lm` | 16384 | 2.710 | 2.588 | 3.308 | 53.11 | 15834 |
| `wp_bpe_npmi` | 16384 | 2.682 | 2.530 | 3.435 | 53.94 | 14635 |
| `morpheme_bpe` | 16384 | 2.633 | 2.529 | 3.395 | 54.94 | 15089 |
| `mdl_bits` | 16384 | 2.439 | 2.370 | 3.991 | 53.56 | 14819 |
| `wp_bpe_scp` | 16384 | 1.995 | 1.641 | 5.184 | 67.43 | 6914 |
| `wordpiece_pmi` | 16384 | 1.026 | 1.017 | 8.900 | 83.23 | 1050 |
| `wp_bpe_ft` | 16384 | 1.026 | 1.017 | 8.900 | 83.23 | 990 |
| `wp_bpe_base` | 16384 | 1.009 | 1.007 | 9.005 | 83.55 | 282 |

Held-out text per language:

| Language | Docs | Bytes | In macro |
| :--- | ---: | ---: | :---: |
| en | 316 | 1,459,019 | yes |
| de | 38 | 92,973 | yes |
| fr | 20 | 88,540 | yes |
| es | 34 | 87,428 | yes |
| ru | 18 | 97,904 | yes |
| zh | 23 | 132,462 | yes |
| ja | 26 | 90,374 | yes |
| ar | 19 | 93,303 | yes |
| hi | 21 | 116,166 | yes |
| hu | 23 | 93,410 | yes |
| fi | 39 | 88,122 | yes |
| python | 18 | 113,050 | yes |
| javascript | 15 | 118,916 | yes |
| java | 24 | 115,673 | yes |
| cpp | 8 | 128,232 | yes |
| go | 11 | 104,232 | yes |
| rust | 13 | 123,740 | yes |
