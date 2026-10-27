# Modular Rust + Python Tokenizers Research & Benchmarking Platform

A high-performance **Rust (`PyO3` / `maturin`) + Python** research and benchmarking platform for designing, modifying, and evaluating subword and dynamic tokenization algorithms across:
1. **Tokka-Bench** (multilingual natural languages + programming languages)
2. **Extended Intrinsic Diagnostics** (Rényi efficiency $E_\alpha$, Morphological Boundary F1, Identifier Boundary F1, Vocabulary Utilization, and Systems Throughput in MB/s)
3. **Extrinsic Proxy-LM Evaluation** (Controlled Llama/GPT-style Transformer trained under a fixed UTF-8 byte budget, evaluated via **Validation Bits-Per-Byte [BPB]** across English, Multilingual, and Code splits locally or on Google Colab GPUs via `colab-cli`).

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
│   └── benchmarks/
│       ├── tokka/                  # Tokka-Bench multilingual + code evaluation suite
│       ├── intrinsic/              # Rényi efficiency, Morphological F1, Identifier F1, Throughput MB/s
│       └── colab/                  # Extrinsic Proxy-LM fixed-byte-budget trainer & Colab GPU orchestrator
├── vendor/tokka-bench/             # Vendored bgub/tokka-bench reference suite
├── benchmarks/results/             # Generated JSON, CSV, and Markdown leaderboards
└── tests/e2e/                      # 209 automated E2E & adversarial tests (Tiers 1–5)
```

---

## Implemented Tokenizer Algorithms & Modifications

| Family | Variant | Objective / Mechanism |
| :--- | :--- | :--- |
| **Bottom-Up Merge** | `bpe_frequency` | Standard Byte-Level BPE: $\text{score}(u, v) = \text{count}(uv)$ |
| **Bottom-Up Merge** | `wordpiece_pmi` | Pointwise Mutual Information / WordPiece likelihood: $\frac{\text{count}(uv)}{\text{count}(u)\text{count}(v)}$ |
| **Bottom-Up Merge** | `mdl_bits` | Minimum Description Length ($\Delta\text{Bits}$): corpus compression gain minus codebook overhead |
| **Bottom-Up Merge** | `length_penalized` | Length-Penalized BPE: $\text{count}(uv) / (|u| + |v|)^\beta$ |
| **Curriculum BPE** | `super_bpe` | **Two-Stage SuperBPE**: enforces whitespace pre-tokenization up to $V_{\text{switch}}$, then lifts whitespace barriers up to $V_{\text{final}}$ |
| **Morphological Prior** | `morpheme_bpe` | **Morpheme-Guided BPE**: scales candidate merge scores by unsupervised/gold morpheme boundary alignment factor $\lambda$ |
| **Top-Down Pruning** | `unigram_lm` | **Unigram Language Model**: suffix-seeded trie + log-space forward-backward EM and Viterbi loss pruning |
| **Stochastic / Dynamic** | `bpe_dropout` | **BPE-Dropout**: randomly skips eligible merges with probability $p$ during LM training |
| **Stochastic / Dynamic** | `unigram_sampling` | **Subword Regularization**: Forward-Filtering Backward-Sampling (FFBS) & $k$-best Viterbi lattice sampling |
| **Stochastic / Dynamic** | `byte_entropy` | **Byte-Entropy Dynamic Patching**: places patch boundaries where Markov byte transition entropy exceeds threshold $\tau$ |

---

## Quickstart & CLI Usage

### 1. Build & Run Tests
```bash
# Run native Rust test suite (54 tests)
cargo test --workspace

# Build PyO3 extension in release mode & run full Python E2E suite (209 tests)
uv run maturin develop --release
uv run pytest tests/e2e/ -v
```

### 2. Prepare Standardized Corpus
```bash
# Offline deterministic smoke corpus (or add --stream-remote for live HuggingFace streaming)
uv run python -m tokenisers.data --preset smoke --force-offline
```

### 3. Run Benchmarks
```bash
# Tier 1: Tokka-Bench Multilingual & Code Matrix
uv run python -m tokenisers.benchmarks.tokka --vocab-size 400 --output-dir benchmarks/results/tokka

# Tier 2: Extended Intrinsic Suite (Rényi Efficiency, Morph-F1, Identifier-F1, Throughput)
uv run python -m tokenisers.benchmarks.intrinsic --vocab-size 400 --output-dir benchmarks/results/intrinsic

# Tier 3: Extrinsic Proxy-LM Fixed-Byte-Budget Benchmark (Google Colab GPU or Local)
uv run python -m tokenisers.benchmarks.colab --vocab-size 400 --byte-budget 150000 --gpu T4 --output-dir benchmarks/results/colab
```

---

## Empirical Benchmark Leaderboards (`V = 400` Smoke Sweep)

### 1. Tokka-Bench Multilingual (10 Natural Languages) + Code Matrix
Full artifacts: [`benchmarks/results/tokka/`](benchmarks/results/tokka/)

| Tokenizer Variant | Macro Bytes/Token (↑) | Macro Fertility (↓) | Continued Word Rate % (↓) | Total Unique Tokens (↑) |
| :--- | ---: | ---: | ---: | ---: |
| `wordpiece_pmi` | 2.000 | 6.221 | 88.57% | 345 |
| `unigram_lm` | 1.584 | 4.356 | 82.62% | 483 |
| `super_bpe` | 1.520 | 4.952 | 90.77% | 531 |
| `bpe_frequency` | 1.439 | 4.616 | 86.99% | 508 |
| `mdl_bits` | 1.438 | 4.765 | 86.01% | 432 |
| `length_penalized` | 1.405 | 4.917 | 86.69% | 492 |
| `morpheme_bpe` | 1.402 | 4.723 | 88.50% | 508 |

### 2. Extended Intrinsic Evaluation Leaderboard
Full artifacts: [`benchmarks/results/intrinsic/`](benchmarks/results/intrinsic/)

| Tokenizer | Bytes/Token (↑) | Fertility (↓) | Rényi $E_{1.0}$ | Rényi $E_{2.0}$ | Rényi $E_{2.5}$ | Morph F1 (↑) | Ident F1 (↑) | Encode (MB/s) | Decode (MB/s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 1.827 | 4.789 | **0.879** | **0.822** | **0.794** | 0.468 | 0.397 | 0.1 | 54.4 |
| `morpheme_bpe` | 1.586 | 4.608 | 0.799 | 0.588 | 0.515 | **0.503** | **0.477** | 0.1 | 55.5 |
| `bpe_frequency` | 1.667 | 4.331 | 0.801 | 0.577 | 0.502 | 0.442 | 0.419 | 0.1 | 58.8 |
| `length_penalized` | 1.632 | 4.448 | 0.781 | 0.577 | 0.506 | 0.423 | 0.442 | 0.1 | 49.3 |
| `mdl_bits` | 1.514 | 4.773 | 0.724 | 0.543 | 0.486 | 0.414 | 0.398 | 0.1 | 60.3 |
| `unigram_lm` | 1.810 | **3.879** | 0.779 | 0.541 | 0.471 | 0.102 | 0.429 | **23.4** | **77.2** |
| `wordpiece_pmi` | 1.072 | 7.230 | 0.635 | 0.539 | 0.513 | 0.329 | 0.340 | 0.4 | 60.4 |

### 3. Extrinsic Proxy-LM Validation Bits-Per-Byte (BPB) Leaderboard
Full artifacts: [`benchmarks/results/colab/`](benchmarks/results/colab/) (`150,000` fixed UTF-8 byte training budget per tokenizer)

| Tokenizer Variant | Train Bytes/Token | Tokens Trained (FLOPs proxy) | English BPB (↓) | Multilingual BPB (↓) | Code BPB (↓) | Macro Val BPB (↓) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 1.840 | 81,920 | **1.938** | 2.277 | 3.502 | **2.572** |
| `unigram_lm` | 1.808 | 82,944 | 2.204 | **2.268** | **3.388** | **2.620** |
| `bpe_frequency` | 1.674 | 90,112 | 2.207 | 2.590 | 3.417 | **2.738** |
| `bpe_dropout_p0.1` | 1.673 | 90,112 | 2.228 | 2.615 | 3.516 | **2.786** |
| `morpheme_bpe` | 1.622 | 92,672 | 2.331 | 2.300 | 3.749 | **2.793** |
| `length_penalized` | 1.649 | 91,136 | 2.197 | 2.602 | 3.614 | **2.804** |
| `mdl_bits` | 1.506 | 99,840 | 2.980 | 2.435 | 3.962 | **3.126** |
| `wordpiece_pmi` | 1.066 | 140,800 | 3.791 | 2.680 | 4.231 | **3.567** |
