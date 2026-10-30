- Colab CPU runtime (no accelerator requested), session `tokenisers-throughput-d8a4e8da`; 2 vCPUs (1 core(s) x 2 threads, KVM VM): AMD EPYC 7B12, 12.7 GiB RAM, Python 3.13.18, glibc 2.39; extension built on the VM from commit `893cb9b516` (rustc 1.99.1 (58cf8f9c3 2026-10-13), 66.4 s); commit `893cb9b516`; generated 2026-10-30T15:11:02Z
- 39 tokenizers: the suite's variants at V = 16384, 32768, 65536, retrained from the real train split (52.5 MB, dataset `hf-fb5b8df7af00b10f`), saved, reloaded on the measuring machine and checked there: identical token ids (sha256) on every throughput document
- 666 documents, 3.1 MB, one encode/decode call per document; each cell is the median of 5 timed passes after 1 warm-up (`IntrinsicEvaluator.benchmark_throughput`), measured in 2 full runs over all 39 tokenizers: run 1 / run 2
- run-to-run spread (max/min over the runs, per tokenizer): encode median 1.03x, max 1.50x; decode median 1.02x, max 1.52x
- 1-min load average (the benchmark's own process adds about 1) and MemAvailable before -> after each run: run 1: 1.37 -> 1.26, 11.3 GiB -> 11.3 GiB (66 s); run 2: 1.26 -> 1.25, 11.3 GiB -> 11.3 GiB (66 s)

| Tokenizer | \|V\| | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: |
| `bpe_frequency` | 16384 | 14.39 / 14.15 | 65.90 / 66.52 |
| `wordpiece_pmi` | 16384 | 35.48 / 36.88 | 48.04 / 47.73 |
| `wp_bpe_base` | 16384 | 38.69 / 39.08 | 46.82 / 48.48 |
| `wp_bpe_ft` | 16384 | 24.55 / 36.79 | 30.62 / 46.61 |
| `wp_bpe_pig` | 16384 | 14.97 / 14.55 | 66.32 / 63.00 |
| `wp_bpe_scp` | 16384 | 20.80 / 20.34 | 47.43 / 47.23 |
| `wp_bpe_npmi` | 16384 | 15.63 / 11.33 | 64.84 / 46.72 |
| `wp_bpe_ppig` | 16384 | 14.66 / 14.97 | 65.57 / 66.12 |
| `mdl_bits` | 16384 | 15.05 / 14.21 | 62.72 / 60.63 |
| `length_penalized` | 16384 | 14.16 / 14.47 | 67.87 / 68.87 |
| `super_bpe` | 16384 | 5.44 / 7.30 | 92.39 / 90.95 |
| `morpheme_bpe` | 16384 | 14.09 / 14.58 | 62.14 / 62.77 |
| `unigram_lm` | 16384 | 9.28 / 6.54 | 64.88 / 47.82 |
| `bpe_frequency` | 32768 | 12.78 / 12.97 | 65.32 / 59.56 |
| `wordpiece_pmi` | 32768 | 36.02 / 36.15 | 47.24 / 46.26 |
| `wp_bpe_base` | 32768 | 38.13 / 37.78 | 47.07 / 47.52 |
| `wp_bpe_ft` | 32768 | 36.15 / 37.04 | 47.72 / 46.54 |
| `wp_bpe_pig` | 32768 | 13.56 / 13.45 | 66.57 / 63.72 |
| `wp_bpe_scp` | 32768 | 12.92 / 18.14 | 38.80 / 50.04 |
| `wp_bpe_npmi` | 32768 | 14.25 / 14.01 | 63.76 / 65.25 |
| `wp_bpe_ppig` | 32768 | 13.48 / 13.48 | 64.17 / 65.21 |
| `mdl_bits` | 32768 | 13.99 / 14.14 | 63.58 / 64.44 |
| `length_penalized` | 32768 | 13.67 / 12.99 | 70.47 / 62.74 |
| `super_bpe` | 32768 | 6.24 / 6.41 | 93.52 / 96.24 |
| `morpheme_bpe` | 32768 | 10.06 / 13.54 | 49.54 / 66.11 |
| `unigram_lm` | 32768 | 8.17 / 8.17 | 64.73 / 63.98 |
| `bpe_frequency` | 65536 | 11.43 / 8.80 | 66.36 / 45.52 |
| `wordpiece_pmi` | 65536 | 34.74 / 34.77 | 47.86 / 46.00 |
| `wp_bpe_base` | 65536 | 36.69 / 38.01 | 46.20 / 47.92 |
| `wp_bpe_ft` | 65536 | 35.50 / 35.92 | 46.47 / 46.12 |
| `wp_bpe_pig` | 65536 | 12.38 / 12.11 | 63.29 / 64.46 |
| `wp_bpe_scp` | 65536 | 16.16 / 15.60 | 57.87 / 57.56 |
| `wp_bpe_npmi` | 65536 | 9.66 / 13.08 | 63.66 / 63.99 |
| `wp_bpe_ppig` | 65536 | 11.75 / 11.57 | 62.24 / 61.81 |
| `mdl_bits` | 65536 | 12.53 / 9.10 | 61.22 / 45.02 |
| `length_penalized` | 65536 | 12.00 / 11.72 | 64.44 / 63.10 |
| `super_bpe` | 65536 | 5.46 / 5.25 | 85.93 / 87.81 |
| `morpheme_bpe` | 65536 | 9.08 / 11.69 | 46.09 / 62.81 |
| `unigram_lm` | 65536 | 7.06 / 6.73 | 66.36 / 65.75 |
