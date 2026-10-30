- requested V = 65536; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `b8f3504b3e`; generated 2026-10-30T07:25:19Z
- arguments: `--train-bytes 10000000 --vocab-size 65536 --output-dir benchmarks/results/v65536/tokka`
- macro = unweighted mean over the 15 languages with >= 3 held-out documents and >= 10,000 bytes (en, de, fr, es, ru, zh, ar, hi, hu, fi, python, javascript, java, cpp, go); micro = totals summed over all 17 languages (token-weighted); unique tokens = union over languages

| Tokenizer | \|V\| | Macro bytes/token (↑) | Micro bytes/token (↑) | Macro fertility (↓) | Macro continued-token % (↓) | Unique tokens used |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 65536 | 4.238 | 4.390 | 2.182 | 47.05 | 26837 |
| `length_penalized` | 65536 | 3.141 | 2.950 | 2.941 | 46.23 | 23283 |
| `bpe_frequency` | 65536 | 3.132 | 2.943 | 2.950 | 46.42 | 22853 |
| `morpheme_bpe` | 65536 | 3.094 | 2.906 | 2.986 | 47.30 | 21840 |
| `wp_bpe_ppig` | 65536 | 3.066 | 2.887 | 3.037 | 47.46 | 20872 |
| `wp_bpe_pig` | 65536 | 3.062 | 2.873 | 3.037 | 47.96 | 20236 |
| `unigram_lm` | 65536 | 2.972 | 2.827 | 3.103 | 48.48 | 21495 |
| `wp_bpe_npmi` | 65536 | 2.911 | 2.756 | 3.297 | 49.38 | 18361 |
| `wp_bpe_scp` | 65536 | 2.784 | 2.512 | 3.468 | 54.95 | 13014 |
| `mdl_bits` | 40496 | 2.625 | 2.508 | 3.878 | 50.73 | 17288 |
| `wordpiece_pmi` | 65536 | 1.241 | 1.056 | 8.579 | 80.73 | 3329 |
| `wp_bpe_ft` | 65536 | 1.233 | 1.053 | 8.587 | 80.85 | 3087 |
| `wp_bpe_base` | 65536 | 1.022 | 1.013 | 9.222 | 84.28 | 428 |

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
