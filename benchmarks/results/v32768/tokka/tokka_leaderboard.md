- requested V = 32768; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T13:02:20Z
- arguments: `--vocab-size 32768 --output-dir benchmarks/results/v32768/tokka`
- macro = unweighted mean over the 17 languages with >= 3 held-out documents and >= 10,000 bytes (en, de, fr, es, ru, zh, ja, ar, hi, hu, fi, python, javascript, java, cpp, go, rust); micro = totals summed over all 17 languages (token-weighted); unique tokens = union over languages

| Tokenizer | \|V\| | Macro bytes/token (↑) | Micro bytes/token (↑) | Macro fertility (↓) | Macro continued-token % (↓) | Unique tokens used |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 32768 | 3.891 | 4.062 | 2.306 | 48.11 | 29931 |
| `bpe_frequency` | 32768 | 3.047 | 2.838 | 2.980 | 47.34 | 29164 |
| `length_penalized` | 32768 | 3.040 | 2.836 | 2.985 | 47.27 | 29291 |
| `wp_bpe_ppig` | 32768 | 3.017 | 2.804 | 3.032 | 47.79 | 27605 |
| `wp_bpe_pig` | 32768 | 3.005 | 2.790 | 3.049 | 48.17 | 27032 |
| `wp_bpe_npmi` | 32768 | 2.932 | 2.719 | 3.159 | 49.41 | 25491 |
| `morpheme_bpe` | 32768 | 2.925 | 2.747 | 3.088 | 49.34 | 26359 |
| `unigram_lm` | 32768 | 2.921 | 2.752 | 3.089 | 48.82 | 28094 |
| `mdl_bits` | 32768 | 2.662 | 2.535 | 3.717 | 48.78 | 26396 |
| `wp_bpe_scp` | 32768 | 2.256 | 1.884 | 4.548 | 63.70 | 10181 |
| `wordpiece_pmi` | 32768 | 1.032 | 1.020 | 8.889 | 83.13 | 1784 |
| `wp_bpe_ft` | 32768 | 1.031 | 1.020 | 8.890 | 83.15 | 1693 |
| `wp_bpe_base` | 32768 | 1.015 | 1.011 | 8.949 | 83.53 | 383 |

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
