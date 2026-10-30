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
_No results committed (`benchmarks/results/tokka/tokka_matrix.json` is missing). Run `uv run python -m tokenisers.benchmarks.tokka`, then `uv run python scripts/gen_readme_tables.py`._
<!-- END GENERATED: tokka -->

### 2. Extended Intrinsic Evaluation
<!-- BEGIN GENERATED: intrinsic -->
_No results committed (`benchmarks/results/intrinsic/intrinsic_leaderboard.json` is missing). Run `uv run python -m tokenisers.benchmarks.intrinsic`, then `uv run python scripts/gen_readme_tables.py`._
<!-- END GENERATED: intrinsic -->

### 3. Extrinsic Proxy-LM Validation Bits per Byte
<!-- BEGIN GENERATED: colab -->
_No results committed (`benchmarks/results/colab/extrinsic_bpb_results.json` is missing). Run `uv run python -m tokenisers.benchmarks.colab`, then `uv run python scripts/gen_readme_tables.py`._
<!-- END GENERATED: colab -->
