- requested V = 65536; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T13:06:59Z
- arguments: `--skip-memory-check --vocab-size 65536 --output-dir benchmarks/results/v65536/tokka`
- macro = unweighted mean over the 17 languages with >= 3 held-out documents and >= 10,000 bytes (en, de, fr, es, ru, zh, ja, ar, hi, hu, fi, python, javascript, java, cpp, go, rust); micro = totals summed over all 17 languages (token-weighted); unique tokens = union over languages

| Tokenizer | \|V\| | Macro bytes/token (↑) | Micro bytes/token (↑) | Macro fertility (↓) | Macro continued-token % (↓) | Unique tokens used |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 65536 | 4.369 | 4.537 | 2.058 | 43.42 | 50752 |
| `bpe_frequency` | 65536 | 3.291 | 3.007 | 2.777 | 42.71 | 47483 |
| `length_penalized` | 65536 | 3.285 | 3.006 | 2.780 | 42.81 | 48001 |
| `wp_bpe_ppig` | 65536 | 3.266 | 2.975 | 2.818 | 43.26 | 44082 |
| `wp_bpe_pig` | 65536 | 3.258 | 2.963 | 2.828 | 43.62 | 42903 |
| `morpheme_bpe` | 65536 | 3.228 | 2.956 | 2.826 | 43.91 | 43824 |
| `wp_bpe_npmi` | 65536 | 3.170 | 2.887 | 2.938 | 44.94 | 39469 |
| `unigram_lm` | 65536 | 3.122 | 2.897 | 2.912 | 44.55 | 42909 |
| `mdl_bits` | 65536 | 2.882 | 2.686 | 3.485 | 43.87 | 42098 |
| `wp_bpe_scp` | 65536 | 2.652 | 2.253 | 3.805 | 57.18 | 17737 |
| `wordpiece_pmi` | 65536 | 1.043 | 1.025 | 8.865 | 83.06 | 3609 |
| `wp_bpe_ft` | 65536 | 1.037 | 1.022 | 8.874 | 83.08 | 2649 |
| `wp_bpe_base` | 65536 | 1.015 | 1.011 | 8.948 | 83.53 | 439 |

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
