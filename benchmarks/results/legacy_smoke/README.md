# Legacy smoke results — invalid, kept only for reference

These files were produced by the benchmark code at commit `ef08acb` and were shown in README.md
before the review fixes. Do not cite them. They are wrong for these reasons (review findings B1–B18):

- **Synthetic data only.** Every CLI ignored the prepared corpus and trained on the 120 KB offline
  synthetic sample (B1).
- **No held-out evaluation.** Tokka scored the training sentences; the intrinsic suite scored
  `corpus[:80]`; the proxy LM's validation text was also in its training text and in the tokenizer
  corpus (B2). Morphology gold words were added 15× to the tokenizer corpus and used as the
  morpheme_bpe lexicon (B6).
- **Tokka numbers are not tokka-bench's.** Each "language" was one pangram repeated 5× with no
  separator (zh = 90 bytes/token), unique tokens were summed instead of unioned, and fertility /
  continued-word rate used other definitions (B3, B7, B8). The files say V=512; README said 400 (B4).
- **Intrinsic bugs.** Unigram morph F1 used token-id string lengths as offsets (0.102 instead of
  0.409, B5); throughput was one timed run floored at 0.1 MB/s (B9); Rényi values were clamped (B16);
  identifier F1 used 5 hand-picked identifiers (B17).
- **Proxy-LM provenance is false.** The JSON says `"hardware": "CPU"`: the Colab call silently fell
  back to local CPU, while README said Colab GPU (B11). The LM cycled 4.7 KB of text ~32×, one seed,
  N(0,1) embedding init with a step-1 loss of ~115 nats (B12, B13, B15, B18).
- **README tables did not match these files** (B10), and several variants had byte-identical
  vocabularies but were listed as separate results.

Current results are written to `benchmarks/results/{tokka,intrinsic,colab}/` by the fixed CLIs and
copied into README.md by `scripts/gen_readme_tables.py`.
