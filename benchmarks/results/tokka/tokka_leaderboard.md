- requested V = 16384; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `df4c7d8edc`; generated 2026-10-30T05:14:40Z
- arguments: `--train-bytes 10000000`
- macro = unweighted mean over the 15 languages with >= 3 held-out documents and >= 10,000 bytes (en, de, fr, es, ru, zh, ar, hi, hu, fi, python, javascript, java, cpp, go); micro = totals summed over all 17 languages (token-weighted); unique tokens = union over languages

| Tokenizer | \|V\| | Macro bytes/token (↑) | Micro bytes/token (↑) | Macro fertility (↓) | Macro continued-token % (↓) | Unique tokens used |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 16384 | 3.401 | 3.559 | 2.706 | 56.25 | 13093 |
| `length_penalized` | 16384 | 2.709 | 2.619 | 3.389 | 55.40 | 12549 |
| `bpe_frequency` | 16384 | 2.706 | 2.620 | 3.391 | 55.57 | 12346 |
| `unigram_lm` | 16384 | 2.649 | 2.574 | 3.450 | 56.06 | 12470 |
| `wp_bpe_pig` | 16384 | 2.625 | 2.544 | 3.533 | 56.57 | 11085 |
| `wp_bpe_ppig` | 16384 | 2.611 | 2.543 | 3.582 | 56.05 | 11442 |
| `morpheme_bpe` | 16384 | 2.600 | 2.533 | 3.515 | 57.57 | 11165 |
| `wp_bpe_npmi` | 16384 | 2.513 | 2.441 | 3.797 | 57.67 | 10182 |
| `mdl_bits` | 16384 | 2.375 | 2.315 | 4.216 | 56.95 | 10865 |
| `wp_bpe_scp` | 16384 | 1.962 | 1.651 | 5.145 | 70.21 | 3616 |
| `wordpiece_pmi` | 16384 | 1.035 | 1.017 | 9.193 | 84.04 | 848 |
| `wp_bpe_ft` | 16384 | 1.034 | 1.016 | 9.195 | 84.05 | 803 |
| `wp_bpe_base` | 16384 | 1.011 | 1.007 | 9.288 | 84.52 | 305 |

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
