# Modular Rust + Python Tokenizers Research & Benchmarking Platform

A high-performance **Rust (`PyO3` / `maturin`) + Python** research and benchmarking platform for designing, modifying, and evaluating subword and dynamic tokenization algorithms across:
1. **Tokka-Bench** (multilingual natural languages + programming languages)
2. **Extended Intrinsic Diagnostics** (Rényi efficiency $E_\alpha$, Morphological Boundary F1, Identifier Boundary F1, Vocabulary Utilization, and Systems Throughput in MB/s)
3. **Extrinsic Proxy-LM Evaluation** (a small causal Transformer trained under a fixed UTF-8 byte budget, evaluated via **Validation Bits-Per-Byte [BPB]** across English, Multilingual, and Code splits locally or on Google Colab GPUs via `colab-cli`). The proxy LM is *not* a Llama/GPT block: it uses PyTorch `TransformerEncoderLayer`s (pre-LayerNorm, ReLU FFN, biases) with a causal mask, tied input/output embeddings and **no positional encoding of any kind** (NoPE: token order reaches the model only through the causal mask). Keep that in mind when comparing its BPB numbers with results from RoPE/SwiGLU models.

---

## Architecture & Repository Layout

```text
├── Cargo.toml                      # Rust workspace manifest
├── pyproject.toml                  # Maturin + uv build & CLI configuration
├── crates/
│   ├── tokenisers_core/            # Pure Rust core library (54/54 unit, integration & stress tests)
│   │   └── src/
│   │       ├── arena.rs            # Doubly-linked symbol arena & inverted pair index
│   │       ├── vocab.rs            # Byte-level lossless base alphabet & vocabulary mapping
│   │       ├── scoring/            # Pluggable scorers: Frequency (BPE), WordPiece/PMI, MDL (ΔBits), Length-Penalized
│   │       ├── engine/             # Unified bottom-up merge engine & Top-Down Unigram EM + Viterbi pruning
│   │       ├── curricula/          # Two-Stage SuperBPE & Morpheme-Guided boundary priors
│   │       ├── inference/          # Greedy, BPE-Dropout, Unigram FFBS/k-best sampling, Byte-Entropy patching
│   │       └── export/             # Canonical HuggingFace tokenizer.json serializer
│   └── tokenisers_py/              # PyO3 C-extension bindings (`tokenisers._core`) with GIL release
├── python/tokenisers/
│   ├── models.py                   # High-level Python API (BpeTokenizer, UnigramTokenizer, SuperBpeTokenizer, MorphemeBpeTokenizer, ByteEntropyTokenizer)
│   ├── wrapper.py                  # DynamicResearchTokenizer (HuggingFace PreTrainedTokenizer subclass)
│   ├── data/                       # Streaming & cached dataset pipeline (FineWeb-Edu, FineWeb-2, StarCoder, SIGMORPHON)
│   ├── dashboard/                  # Artificial Analysis-style interactive Web UI & REST API server (`tokenisers-dashboard`)
│   └── benchmarks/
│       ├── tokka/                  # Tokka-Bench multilingual + code evaluation suite
│       ├── intrinsic/              # Rényi efficiency, Morphological F1, Identifier F1, Throughput MB/s
│       └── colab/                  # Extrinsic Proxy-LM fixed-byte-budget trainer & Colab GPU orchestrator
├── vendor/tokka-bench/             # Vendored bgub/tokka-bench reference suite
├── benchmarks/results/             # Generated JSON, CSV, and Markdown leaderboards
└── tests/e2e/                      # 217 automated E2E & adversarial tests (Tiers 1–5 + WP-BPE + Dashboard)
```

---

## Implemented Tokenizer Algorithms & Modifications

| Family | Variant | Objective / Mechanism |
| :--- | :--- | :--- |
| **Bottom-Up Merge** | `bpe_frequency` | Standard Byte-Level BPE: $\text{score}(u, v) = \text{count}(uv)$ |
| **Bottom-Up Merge** | `wordpiece_pmi` | Laplace-smoothed WordPiece likelihood: $\frac{\text{count}(uv)}{(\text{count}(u)+\alpha)(\text{count}(v)+\alpha)}$, $\alpha = 1$ by default ($\alpha = 0$ with no cutoff is exactly `wp_bpe_base`) |
| **WP-BPE (Gen 1)** | `wp_bpe_base` | **wp-bpe-base**: Uncalibrated PMI BPE merges without frequency cutoff: $\frac{\text{count}(uv)}{\text{count}(u)\text{count}(v)}$ |
| **WP-BPE (Gen 1)** | `wp_bpe_ft` | **wp-bpe-ft**: Frequency-Thresholded PMI BPE merges: same ratio, only for $\text{freq}(uv) \ge 2$ |
| **WP-BPE (Gen 1)** | `wp_bpe_pig` | **wp-bpe-pig**: Pointwise Information Gain BPE: $\text{freq}(uv) \cdot \text{PMI}(u, v)$ |
| **WP-BPE (Gen 2 Improved)** | `wp_bpe_scp` | **wp-bpe-scp**: Symmetric Conditional Probability / Geometric-Mean Calibrated PMI with Kneser-Ney discounting: $\frac{(\text{freq}(uv) - d)_+}{\sqrt{\text{freq}(u)\text{freq}(v)}}$ |
| **WP-BPE (Gen 2 Improved)** | `wp_bpe_npmi` | **wp-bpe-npmi**: Bouma Normalized PMI ($\text{NPMI} \in [-1, 1]$) weighted by sub-linear frequency support $\text{freq}(uv)^\alpha$ |
| **WP-BPE (Gen 2 Improved)** | `wp_bpe_ppig` | **wp-bpe-ppig**: Positive Length-Regularized Pointwise Information Gain: $\frac{\text{freq}(uv)\cdot\max(0, \text{PMI})}{(|u|+|v|)^\eta}$ |
| **Bottom-Up Merge** | `mdl_bits` | Minimum Description Length ($\Delta\text{Bits}$): corpus compression gain minus codebook overhead |
| **Bottom-Up Merge** | `length_penalized` | Length-Penalized BPE: $\text{count}(uv) / (|u| + |v|)^\beta$ |
| **Curriculum BPE** | `super_bpe` | **Two-Stage SuperBPE**: enforces whitespace pre-tokenization up to $V_{\text{switch}}$, then lifts whitespace barriers up to $V_{\text{final}}$ |
| **Morphological Prior** | `morpheme_bpe` | **Morpheme-Guided BPE**: scales each candidate's score by $e^{\lambda M(u,v)}$, where $M \in [-1,1]$ is the share of the pair's current occurrences in lexicon words whose merged token stays inside one morpheme minus the share that would span a boundary; pairs with $\text{count}(uv)\,e^{\lambda M} < 1$ are not merged. The caller supplies the lexicon (training annotations or unsupervised boundaries) |
| **Top-Down Pruning** | `unigram_lm` | **Unigram Language Model**: suffix-seeded trie + log-space forward-backward EM and Viterbi loss pruning |
| **Stochastic / Dynamic** | `bpe_dropout` | **BPE-Dropout**: randomly skips eligible merges with probability $p$ during LM training |
| **Stochastic / Dynamic** | `unigram_sampling` | **Subword Regularization**: Forward-Filtering Backward-Sampling (FFBS) & $k$-best Viterbi lattice sampling |
| **Stochastic / Dynamic** | `byte_entropy` | **Byte-Entropy Dynamic Patching**: places patch boundaries where Markov byte transition entropy exceeds threshold $\tau$ |

---

## Quickstart & CLI Usage

### 1. Build & Run Tests
```bash
# Run native Rust test suite (51+ tests)
cargo test --workspace

# Build PyO3 extension in release mode & run full Python E2E suite (217 tests)
uv run maturin develop --release
uv run pytest tests/e2e/ -v
```

### 2. Launch Interactive Benchmark Dashboard (Artificial Analysis-Style Web UI)
```bash
# Starts local web server at http://127.0.0.1:8765 with interactive charts & live tokenization playground
uv run python -m tokenisers.dashboard --port 8765
```

### 3. Prepare Standardized Corpus
```bash
# Real corpus: FineWeb-Edu + FineWeb-2 (10 languages) + code, de-duplicated, disjoint train/val/test,
# provenance in manifest.json (default ~50 MB train; StarCoder needs HF_TOKEN, else an ungated fallback is recorded)
uv run python -m tokenisers.data --preset small --sigmorphon eng fra hun
# Synthetic offline test corpus (labelled synthetic everywhere)
uv run python -m tokenisers.data --preset smoke --offline
```

### 4. Run Benchmarks
All three tiers read the prepared dataset (step 3): tokenizers train on its **train** split only;
tokka and the intrinsic suite evaluate on the held-out **test** split. The proxy LM trains on
documents of the same **train** split the tokenizers were trained on (shuffled, each used at most
once; *not* held out from tokenizer training) and is scored on the held-out **val** split.
Disjointness of train and evaluation documents is asserted at runtime. The default vocabulary is
16,384 (`--vocab-size 32768` / `65536` for the larger sizes). Every result JSON carries a `meta`
block with the dataset manifest, requested and actual vocabulary sizes, per-variant training
parameters and warnings, seeds, hardware/backend and git commit. Variants that produce
byte-identical tokenizers are detected and reported as one result.

```bash
# Tier 1: Tokka-Bench metrics (vendored tokka-bench definitions) per language of the test split.
# The macro average only includes languages with >= 3 held-out documents and >= 10 KB
# (--min-docs-per-lang / --min-bytes-per-lang); the others are reported and listed with their sizes.
uv run python -m tokenisers.benchmarks.tokka

# Tier 2: Rényi efficiency, utilisation, SIGMORPHON 2022 morph F1 (test split gold; morpheme_bpe
# lexicon from the train split), identifier F1 (mined from held-out code), throughput (median of 5)
uv run python -m tokenisers.benchmarks.intrinsic

# Tier 3: proxy-LM bits per byte, fixed 10 MB train budget, 3 seeds (mean ± sd), on a Colab GPU.
# A Colab failure stops the run unless --allow-local-fallback is given (the fallback is recorded);
# --local runs on this machine. The backend actually used is recorded and shown with the table.
uv run python -m tokenisers.benchmarks.colab --gpu T4

# Copy the result JSON into the tables below
uv run python scripts/gen_readme_tables.py
```

`--offline` runs everything on the synthetic test corpus (labelled synthetic in the output).

**Memory.** Peak training memory grows linearly with the train split and, for the merge-based
variants, with the vocabulary size (measured on 4 MB of the real corpus at V=16k and 64k:
`unigram_lm` 139 bytes per text byte at both sizes, `super_bpe` 78 at 16k and 100 at 64k, the
other BPE variants at most 45). The CLIs budget 175 / 125 / 60 bytes per byte (~25% margin) plus
600 MB, so a 50 MB train split needs ~9 GiB for `unigram_lm`. Before preparing the dataset they
exit if that exceeds the memory the process may use: a cgroup / `systemd-run -p MemoryMax=` limit
in full, otherwise `MemAvailable` minus a reserve of max(2 GiB, 15% of RAM) for the rest of the
machine. Lower `--train-bytes`, drop the variant from `--variants`, or run under an explicit
`MemoryMax=` scope.

---

## Benchmark Results

The tables below are generated from `benchmarks/results/*/` by `scripts/gen_readme_tables.py`; a
test fails if they differ from the committed JSON. The numbers previously shown here came from
synthetic data, evaluated on training text, with a proxy LM that silently ran on a local CPU; they
are kept, marked invalid, in [`benchmarks/results/legacy_smoke/`](benchmarks/results/legacy_smoke/).

The committed results use the default V = 16,384 but a 10 MB train split (`--train-bytes 10000000`):
the default 50 MB split needs ~9 GiB for `unigram_lm`, more than the shared 15 GB machine they were
produced on could spare. The proxy LM ran on that machine's CPU with a 1 MB byte budget. Each
section's header lists the exact arguments, data sizes, backend and commit. 32k / 64k vocabularies
are supported (`--vocab-size 32768` / `65536`); no results are committed for them yet.

### 1. Tokka-Bench (natural languages + code, held-out test split)
<!-- BEGIN GENERATED: tokka -->
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
<!-- END GENERATED: tokka -->

### 2. Extended Intrinsic Evaluation
<!-- BEGIN GENERATED: intrinsic -->
- requested V = 16384; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `df4c7d8edc`; generated 2026-10-30T05:15:19Z
- arguments: `--train-bytes 10000000`
- morph F1: 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)); identifier F1: 1499 identifiers mined from held-out code; throughput: median of 5 passes over 1.1 MB (206 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 16384 | 3.559 | 2.137 | 0.836 | 0.464 | 0.369 | 0.721 | 6.94 | 84.57 |
| `wp_bpe_base` | 16384 | 1.007 | 6.720 | 0.392 | 0.291 | 0.252 | 0.247 | 44.22 | 61.74 |
| `wp_bpe_ft` | 16384 | 1.016 | 6.664 | 0.391 | 0.290 | 0.252 | 0.247 | 43.35 | 62.31 |
| `wordpiece_pmi` | 16384 | 1.017 | 6.662 | 0.391 | 0.289 | 0.252 | 0.247 | 45.21 | 62.20 |
| `wp_bpe_scp` | 16384 | 1.651 | 3.921 | 0.494 | 0.285 | 0.309 | 0.357 | 28.54 | 53.71 |
| `morpheme_bpe` | 16384 | 2.534 | 2.212 | 0.632 | 0.216 | 0.464 | 0.683 | 17.32 | 54.63 |
| `wp_bpe_pig` | 16384 | 2.545 | 2.178 | 0.617 | 0.211 | 0.395 | 0.717 | 21.39 | 66.88 |
| `length_penalized` | 16384 | 2.620 | 2.106 | 0.643 | 0.210 | 0.365 | 0.722 | 19.45 | 64.72 |
| `bpe_frequency` | 16384 | 2.620 | 2.106 | 0.640 | 0.210 | 0.371 | 0.730 | 18.54 | 62.76 |
| `unigram_lm` | 16384 | 2.574 | 2.131 | 0.628 | 0.207 | 0.408 | 0.715 | 10.70 | 56.74 |
| `wp_bpe_npmi` | 16384 | 2.442 | 2.253 | 0.596 | 0.207 | 0.403 | 0.706 | 22.31 | 71.97 |
| `wp_bpe_ppig` | 16384 | 2.544 | 2.156 | 0.616 | 0.206 | 0.398 | 0.729 | 21.80 | 69.89 |
| `mdl_bits` | 16384 | 2.316 | 2.199 | 0.565 | 0.180 | 0.407 | 0.720 | 22.04 | 66.61 |
<!-- END GENERATED: intrinsic -->

### 3. Extrinsic Proxy-LM Validation Bits per Byte
<!-- BEGIN GENERATED: colab -->
- requested V = 16384; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out val split (0.7 MB); commit `df4c7d8edc`; generated 2026-10-30T06:17:32Z
- arguments: `--local --train-bytes 10000000 --byte-budget 1000000`
- backend: local CPU `Intel(R) Core(TM) Ultra 7 155H`
- proxy LM: pre-LayerNorm causal Transformer (torch TransformerEncoderLayer: ReLU FFN, biases), no positional encoding of any kind (NoPE: order comes only from the causal mask), tied input/output embeddings; d_model=128, n_layers=2, n_heads=4, d_ffn=512, max_seq_len=128
- fixed budget 1,000,000 UTF-8 bytes of LM training text per tokenizer, drawn from the same train split the tokenizers were trained on (each document used once); seeds [0, 1, 2] (mean ± sample sd)

| Tokenizer | \|V\| | Train bytes/token | Tokens trained | english BPB (↓) | multilingual BPB (↓) | code BPB (↓) | Macro BPB (↓) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wp_bpe_ppig` | 16384 | 2.691 | 374,784 | 2.735 ± 0.018 | 3.834 ± 0.037 | 3.638 ± 0.005 | **3.402 ± 0.018** |
| `wp_bpe_pig` | 16384 | 2.679 | 374,784 | 2.765 ± 0.036 | 3.832 ± 0.022 | 3.647 ± 0.019 | **3.415 ± 0.011** |
| `bpe_dropout_p0.1` | 16384 | 2.583 | 389,120 | 2.781 ± 0.003 | 3.832 ± 0.027 | 3.632 ± 0.005 | **3.415 ± 0.008** |
| `unigram_lm` | 16384 | 2.641 | 378,880 | 2.753 ± 0.010 | 3.898 ± 0.045 | 3.694 ± 0.035 | **3.448 ± 0.030** |
| `bpe_frequency` | 16384 | 2.731 | 368,640 | 2.768 ± 0.046 | 3.932 ± 0.051 | 3.646 ± 0.015 | **3.449 ± 0.036** |
| `wp_bpe_npmi` | 16384 | 2.604 | 385,024 | 2.797 ± 0.014 | 3.853 ± 0.008 | 3.700 ± 0.009 | **3.450 ± 0.009** |
| `super_bpe` | 16384 | 3.787 | 264,192 | 2.834 ± 0.002 | 3.986 ± 0.037 | 3.597 ± 0.003 | **3.473 ± 0.014** |
| `morpheme_bpe` | 16384 | 2.613 | 382,976 | 2.735 ± 0.029 | 4.006 ± 0.043 | 3.712 ± 0.042 | **3.484 ± 0.038** |
| `length_penalized` | 16384 | 2.725 | 368,640 | 2.801 ± 0.021 | 3.966 ± 0.069 | 3.726 ± 0.031 | **3.498 ± 0.024** |
| `mdl_bits` | 16384 | 2.462 | 403,456 | 2.811 ± 0.065 | 3.883 ± 0.027 | 3.900 ± 0.071 | **3.531 ± 0.012** |
| `wp_bpe_scp` | 16384 | 1.694 | 587,776 | 3.864 ± 0.166 | 3.913 ± 0.062 | 4.468 ± 0.070 | **4.082 ± 0.092** |
| `wp_bpe_ft` | 16384 | 1.019 | 983,040 | 4.121 ± 0.101 | 5.429 ± 0.196 | 4.883 ± 0.076 | **4.811 ± 0.098** |
| `wordpiece_pmi` | 16384 | 1.019 | 983,040 | 4.127 ± 0.044 | 5.568 ± 0.178 | 4.936 ± 0.098 | **4.877 ± 0.104** |
| `wp_bpe_base` | 16384 | 1.009 | 993,280 | 4.176 ± 0.110 | 5.482 ± 0.155 | 5.002 ± 0.195 | **4.887 ± 0.135** |
<!-- END GENERATED: colab -->
