- requested V = 32768; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `b8f3504b3e`; generated 2026-10-30T07:19:54Z
- arguments: `--train-bytes 10000000 --vocab-size 32768 --output-dir benchmarks/results/v32768/tokka`
- macro = unweighted mean over the 15 languages with >= 3 held-out documents and >= 10,000 bytes (en, de, fr, es, ru, zh, ar, hi, hu, fi, python, javascript, java, cpp, go); micro = totals summed over all 17 languages (token-weighted); unique tokens = union over languages

| Tokenizer | \|V\| | Macro bytes/token (↑) | Micro bytes/token (↑) | Macro fertility (↓) | Macro continued-token % (↓) | Unique tokens used |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 32768 | 3.800 | 3.965 | 2.429 | 51.63 | 19977 |
| `length_penalized` | 32768 | 2.931 | 2.796 | 3.141 | 50.61 | 18212 |
| `bpe_frequency` | 32768 | 2.926 | 2.791 | 3.148 | 50.91 | 17928 |
| `morpheme_bpe` | 32768 | 2.867 | 2.743 | 3.208 | 52.01 | 16726 |
| `wp_bpe_pig` | 32768 | 2.859 | 2.722 | 3.246 | 52.02 | 15830 |
| `wp_bpe_ppig` | 32768 | 2.851 | 2.729 | 3.266 | 51.66 | 16307 |
| `unigram_lm` | 32768 | 2.820 | 2.711 | 3.255 | 52.02 | 17415 |
| `wp_bpe_npmi` | 32768 | 2.727 | 2.613 | 3.506 | 53.26 | 14404 |
| `mdl_bits` | 32768 | 2.567 | 2.467 | 3.947 | 52.20 | 15747 |
| `wp_bpe_scp` | 32768 | 2.391 | 2.086 | 4.171 | 62.75 | 7009 |
| `wordpiece_pmi` | 32768 | 1.072 | 1.033 | 9.000 | 83.51 | 1681 |
| `wp_bpe_ft` | 32768 | 1.048 | 1.021 | 9.149 | 83.91 | 1230 |
| `wp_bpe_base` | 32768 | 1.011 | 1.007 | 9.286 | 84.51 | 322 |

Held-out text per language:

| Language | Docs | Bytes | In macro |
| :--- | ---: | ---: | :---: |
| en | 41 | 291,639 | yes |
| de | 7 | 19,537 | yes |
| fr | 6 | 37,576 | yes |
| es | 3 | 21,020 | yes |
| ru | 6 | 22,131 | yes |
| zh | 8 | 17,357 | yes |
| ja | 2 | 16,979 | no |
| ar | 3 | 17,853 | yes |
| hi | 5 | 21,823 | yes |
| hu | 3 | 21,795 | yes |
| fi | 3 | 19,139 | yes |
| python | 3 | 19,258 | yes |
| javascript | 9 | 21,720 | yes |
| java | 7 | 18,710 | yes |
| cpp | 3 | 79,396 | yes |
| go | 3 | 45,801 | yes |
| rust | 2 | 19,178 | no |
