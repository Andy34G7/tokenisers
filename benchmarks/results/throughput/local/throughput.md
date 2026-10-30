- local, pinned to CPU 3 (max 4800 MHz, governor powersave, SMT sibling(s) 3-4): Intel(R) Core(TM) Ultra 7 155H, 15.1 GiB RAM, Python 3.13.17, glibc 2.44; commit `893cb9b516`; generated 2026-10-30T15:06:11Z
- 39 tokenizers: the suite's variants at V = 16384, 32768, 65536, retrained from the real train split (52.5 MB, dataset `hf-fb5b8df7af00b10f`), saved, reloaded on the measuring machine and checked there: identical token ids (sha256) on every throughput document
- 666 documents, 3.1 MB, one encode/decode call per document; each cell is the median of 5 timed passes after 1 warm-up (`IntrinsicEvaluator.benchmark_throughput`), measured in 2 full runs over all 39 tokenizers: run 1 / run 2
- run-to-run spread (max/min over the runs, per tokenizer): encode median 1.02x, max 1.15x; decode median 1.02x, max 1.13x
- 1-min load average (the benchmark's own process adds about 1) and MemAvailable before -> after each run: run 1: 1.86 -> 1.57, 6.0 GiB -> 5.9 GiB (53 s); run 2: 1.57 -> 1.56, 5.9 GiB -> 5.8 GiB (53 s)

| Tokenizer | \|V\| | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: |
| `bpe_frequency` | 16384 | 16.08 / 16.18 | 75.57 / 78.14 |
| `wordpiece_pmi` | 16384 | 41.71 / 43.01 | 63.88 / 65.44 |
| `wp_bpe_base` | 16384 | 43.28 / 44.77 | 64.86 / 65.34 |
| `wp_bpe_ft` | 16384 | 41.31 / 41.12 | 63.51 / 63.65 |
| `wp_bpe_pig` | 16384 | 16.78 / 16.33 | 73.35 / 69.56 |
| `wp_bpe_scp` | 16384 | 21.60 / 22.21 | 48.54 / 49.26 |
| `wp_bpe_npmi` | 16384 | 16.30 / 17.48 | 67.04 / 72.23 |
| `wp_bpe_ppig` | 16384 | 16.55 / 16.82 | 75.11 / 73.96 |
| `mdl_bits` | 16384 | 17.02 / 16.70 | 69.64 / 67.06 |
| `length_penalized` | 16384 | 15.98 / 16.16 | 75.95 / 75.41 |
| `super_bpe` | 16384 | 8.55 / 8.56 | 108.64 / 102.47 |
| `morpheme_bpe` | 16384 | 16.44 / 16.30 | 69.63 / 69.21 |
| `unigram_lm` | 16384 | 10.88 / 11.29 | 67.48 / 70.35 |
| `bpe_frequency` | 32768 | 15.52 / 15.71 | 78.63 / 77.56 |
| `wordpiece_pmi` | 32768 | 42.25 / 43.24 | 64.62 / 65.47 |
| `wp_bpe_base` | 32768 | 43.80 / 42.64 | 66.14 / 65.07 |
| `wp_bpe_ft` | 32768 | 42.25 / 41.89 | 64.73 / 65.12 |
| `wp_bpe_pig` | 32768 | 16.00 / 16.33 | 75.20 / 76.97 |
| `wp_bpe_scp` | 32768 | 19.82 / 20.07 | 51.93 / 52.44 |
| `wp_bpe_npmi` | 32768 | 16.47 / 16.56 | 74.23 / 72.79 |
| `wp_bpe_ppig` | 32768 | 15.77 / 15.75 | 75.78 / 75.06 |
| `mdl_bits` | 32768 | 15.67 / 14.70 | 70.51 / 64.90 |
| `length_penalized` | 32768 | 15.41 / 15.10 | 77.03 / 75.58 |
| `super_bpe` | 32768 | 7.69 / 7.81 | 108.73 / 113.49 |
| `morpheme_bpe` | 32768 | 15.58 / 15.94 | 75.68 / 77.21 |
| `unigram_lm` | 32768 | 9.79 / 10.00 | 70.34 / 70.32 |
| `bpe_frequency` | 65536 | 13.04 / 14.89 | 70.60 / 78.10 |
| `wordpiece_pmi` | 65536 | 38.56 / 41.05 | 61.80 / 64.37 |
| `wp_bpe_base` | 65536 | 42.22 / 39.35 | 63.49 / 59.32 |
| `wp_bpe_ft` | 65536 | 41.04 / 36.91 | 63.06 / 58.94 |
| `wp_bpe_pig` | 65536 | 15.40 / 15.41 | 77.58 / 75.13 |
| `wp_bpe_scp` | 65536 | 18.21 / 18.22 | 61.59 / 61.27 |
| `wp_bpe_npmi` | 65536 | 15.36 / 16.04 | 75.50 / 75.75 |
| `wp_bpe_ppig` | 65536 | 15.03 / 15.10 | 77.64 / 78.24 |
| `mdl_bits` | 65536 | 15.60 / 13.52 | 73.84 / 65.48 |
| `length_penalized` | 65536 | 14.57 / 12.87 | 76.68 / 70.16 |
| `super_bpe` | 65536 | 7.17 / 7.21 | 113.90 / 117.59 |
| `morpheme_bpe` | 65536 | 15.01 / 14.88 | 78.21 / 76.85 |
| `unigram_lm` | 65536 | 8.86 / 8.73 | 74.76 / 74.15 |
