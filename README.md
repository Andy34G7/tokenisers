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

**Colab.** Tier 3 on Colab needs the `colab` CLI (`google-colab-cli`) **>= 0.7.5**: 0.6.0's `exec`
is broken. The run checks `colab version` before it starts a session; upgrade with
`uv tool upgrade google-colab-cli`. The job is not one long `colab exec`: the benchmark uploads the
job, starts it detached on the VM (output in `/content/job.log`) and polls it every
`--poll-interval` seconds (default 60) with short `colab exec` calls. Polls that hang or lose the
connection are retried, and so are the uploads and the final download of `result.json` (5
consecutive failures of a step abort the run). `colab exec` exits 0 even when the remote
code raises, so success is read from the `status.json` the job writes last. A remote exception
fails the run with the remote traceback; so does a job that dies without a status (log tail
shown) or runs past `--timeout`. The session is always stopped.

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

The committed results use a 10 MB train split (`--train-bytes 10000000`): the default 50 MB split
needs ~9 GiB for `unigram_lm`, more than the shared 15 GB machine they were produced on could
spare. Tokka and intrinsic results exist for the default V = 16,384 (sections 1 and 2) and for
V = 32,768 and 65,536 (section 4). The proxy LM (section 3) ran only at V = 16,384, on that
machine's CPU with a 1 MB byte budget, at an older commit than the other tables; a Colab GPU run
with the default 10 MB budget has not been done yet. Each section's header lists the exact
arguments, data sizes, backend and commit.

### 1. Tokka-Bench (natural languages + code, held-out test split)
<!-- BEGIN GENERATED: tokka -->
- requested V = 16384; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `b8f3504b3e`; generated 2026-10-30T07:17:31Z
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
- requested V = 16384; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `b8f3504b3e`; generated 2026-10-30T07:18:20Z
- arguments: `--train-bytes 10000000`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 1499 identifiers mined from held-out code; throughput: median of 5 passes over 1.1 MB (206 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 16384 | 3.559 | 2.137 | 0.836 | 0.464 | 0.273 | 0.721 | 6.16 | 52.41 |
| `wp_bpe_base` | 16384 | 1.007 | 6.720 | 0.392 | 0.291 | 0.206 | 0.247 | 31.21 | 44.80 |
| `wp_bpe_ft` | 16384 | 1.016 | 6.664 | 0.391 | 0.290 | 0.206 | 0.247 | 30.26 | 43.87 |
| `wordpiece_pmi` | 16384 | 1.017 | 6.662 | 0.391 | 0.289 | 0.206 | 0.247 | 30.14 | 45.36 |
| `wp_bpe_scp` | 16384 | 1.651 | 3.921 | 0.494 | 0.285 | 0.249 | 0.357 | 16.66 | 33.24 |
| `morpheme_bpe` | 16384 | 2.534 | 2.212 | 0.632 | 0.216 | 0.404 | 0.683 | 12.40 | 40.38 |
| `wp_bpe_pig` | 16384 | 2.545 | 2.178 | 0.617 | 0.211 | 0.295 | 0.717 | 12.71 | 41.22 |
| `length_penalized` | 16384 | 2.620 | 2.106 | 0.643 | 0.210 | 0.269 | 0.722 | 12.46 | 40.39 |
| `bpe_frequency` | 16384 | 2.620 | 2.106 | 0.640 | 0.210 | 0.273 | 0.730 | 12.35 | 40.73 |
| `unigram_lm` | 16384 | 2.574 | 2.131 | 0.628 | 0.207 | 0.315 | 0.715 | 7.95 | 40.57 |
| `wp_bpe_npmi` | 16384 | 2.442 | 2.253 | 0.596 | 0.207 | 0.307 | 0.706 | 13.35 | 39.98 |
| `wp_bpe_ppig` | 16384 | 2.544 | 2.156 | 0.616 | 0.206 | 0.292 | 0.729 | 12.93 | 40.46 |
| `mdl_bits` | 16384 | 2.316 | 2.199 | 0.565 | 0.180 | 0.315 | 0.720 | 13.18 | 40.16 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.303 | 0.154 | 0.481 | 0.163 | 0.267 | 0.369 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wordpiece_pmi` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_scp` | 0.262 | 0.180 | 0.382 | 0.179 | 0.242 | 0.309 |
| `morpheme_bpe` | 0.475 | 0.326 | 0.504 | 0.357 | 0.359 | 0.464 |
| `wp_bpe_pig` | 0.334 | 0.179 | 0.508 | 0.159 | 0.294 | 0.395 |
| `length_penalized` | 0.296 | 0.147 | 0.477 | 0.162 | 0.264 | 0.365 |
| `bpe_frequency` | 0.302 | 0.154 | 0.486 | 0.160 | 0.265 | 0.371 |
| `unigram_lm` | 0.389 | 0.222 | 0.490 | 0.209 | 0.266 | 0.408 |
| `wp_bpe_npmi` | 0.364 | 0.205 | 0.498 | 0.170 | 0.298 | 0.403 |
| `wp_bpe_ppig` | 0.334 | 0.173 | 0.516 | 0.160 | 0.280 | 0.398 |
| `mdl_bits` | 0.386 | 0.227 | 0.490 | 0.170 | 0.301 | 0.407 |
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

### 4. Larger Vocabularies (V = 32,768 and 65,536)

Same train and test data as sections 1 and 2; only `--vocab-size` differs. The proxy LM was not run at
these sizes.

<details><summary>Tokka-Bench, V = 32,768</summary>

<!-- BEGIN GENERATED: tokka_v32768 -->
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
<!-- END GENERATED: tokka_v32768 -->

</details>

<details><summary>Extended intrinsic evaluation, V = 32,768</summary>

<!-- BEGIN GENERATED: intrinsic_v32768 -->
- requested V = 32768; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `b8f3504b3e`; generated 2026-10-30T07:21:06Z
- arguments: `--train-bytes 10000000 --vocab-size 32768 --output-dir benchmarks/results/v32768/intrinsic`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 1499 identifiers mined from held-out code; throughput: median of 5 passes over 1.1 MB (206 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 32768 | 3.965 | 1.942 | 0.814 | 0.441 | 0.275 | 0.760 | 5.29 | 51.34 |
| `wp_bpe_base` | 32768 | 1.007 | 6.718 | 0.366 | 0.272 | 0.206 | 0.247 | 30.98 | 44.78 |
| `wp_bpe_ft` | 32768 | 1.021 | 6.627 | 0.364 | 0.269 | 0.206 | 0.247 | 29.45 | 43.11 |
| `wordpiece_pmi` | 32768 | 1.033 | 6.537 | 0.363 | 0.268 | 0.214 | 0.247 | 27.81 | 42.63 |
| `wp_bpe_scp` | 32768 | 2.087 | 2.897 | 0.514 | 0.231 | 0.269 | 0.497 | 13.80 | 34.23 |
| `morpheme_bpe` | 32768 | 2.743 | 1.967 | 0.596 | 0.189 | 0.411 | 0.754 | 8.90 | 29.96 |
| `wp_bpe_pig` | 32768 | 2.722 | 1.979 | 0.583 | 0.187 | 0.299 | 0.776 | 11.72 | 38.51 |
| `bpe_frequency` | 32768 | 2.791 | 1.916 | 0.602 | 0.186 | 0.277 | 0.770 | 11.15 | 40.02 |
| `length_penalized` | 32768 | 2.796 | 1.911 | 0.605 | 0.186 | 0.274 | 0.771 | 10.74 | 37.24 |
| `wp_bpe_ppig` | 32768 | 2.730 | 1.962 | 0.584 | 0.185 | 0.294 | 0.782 | 11.54 | 38.32 |
| `wp_bpe_npmi` | 32768 | 2.613 | 2.050 | 0.564 | 0.185 | 0.312 | 0.748 | 11.83 | 38.84 |
| `unigram_lm` | 32768 | 2.711 | 1.960 | 0.586 | 0.183 | 0.321 | 0.768 | 4.53 | 27.46 |
| `mdl_bits` | 32768 | 2.468 | 1.982 | 0.531 | 0.157 | 0.316 | 0.784 | 11.74 | 37.69 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.309 | 0.153 | 0.501 | 0.157 | 0.258 | 0.380 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wordpiece_pmi` | 0.212 | 0.193 | 0.344 | 0.126 | 0.196 | 0.268 |
| `wp_bpe_scp` | 0.314 | 0.194 | 0.426 | 0.169 | 0.241 | 0.346 |
| `morpheme_bpe` | 0.473 | 0.325 | 0.522 | 0.365 | 0.367 | 0.474 |
| `wp_bpe_pig` | 0.344 | 0.186 | 0.530 | 0.156 | 0.279 | 0.408 |
| `bpe_frequency` | 0.310 | 0.155 | 0.503 | 0.156 | 0.260 | 0.381 |
| `length_penalized` | 0.305 | 0.152 | 0.497 | 0.159 | 0.256 | 0.376 |
| `wp_bpe_ppig` | 0.342 | 0.177 | 0.532 | 0.158 | 0.260 | 0.407 |
| `wp_bpe_npmi` | 0.376 | 0.208 | 0.522 | 0.163 | 0.289 | 0.417 |
| `unigram_lm` | 0.398 | 0.231 | 0.514 | 0.220 | 0.243 | 0.421 |
| `mdl_bits` | 0.397 | 0.230 | 0.507 | 0.165 | 0.279 | 0.417 |
<!-- END GENERATED: intrinsic_v32768 -->

</details>

<details><summary>Tokka-Bench, V = 65,536</summary>

<!-- BEGIN GENERATED: tokka_v65536 -->
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
<!-- END GENERATED: tokka_v65536 -->

</details>

<details><summary>Extended intrinsic evaluation, V = 65,536</summary>

<!-- BEGIN GENERATED: intrinsic_v65536 -->
- requested V = 65536; real corpus: tokenizers trained on the train split (10.1 MB); evaluated on the held-out test split (0.7 MB); commit `b8f3504b3e`; generated 2026-10-30T07:27:03Z
- arguments: `--train-bytes 10000000 --vocab-size 65536 --output-dir benchmarks/results/v65536/intrinsic`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 1499 identifiers mined from held-out code; throughput: median of 5 passes over 1.1 MB (206 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 65536 | 4.390 | 1.783 | 0.790 | 0.418 | 0.278 | 0.806 | 7.28 | 90.77 |
| `wp_bpe_base` | 65536 | 1.013 | 6.692 | 0.343 | 0.254 | 0.206 | 0.247 | 43.80 | 65.83 |
| `wp_bpe_ft` | 65536 | 1.053 | 6.388 | 0.338 | 0.248 | 0.214 | 0.247 | 38.69 | 61.77 |
| `wordpiece_pmi` | 65536 | 1.056 | 6.362 | 0.337 | 0.248 | 0.214 | 0.247 | 37.80 | 62.41 |
| `wp_bpe_scp` | 65536 | 2.513 | 2.238 | 0.515 | 0.189 | 0.309 | 0.666 | 17.62 | 59.34 |
| `wp_bpe_pig` | 65536 | 2.873 | 1.826 | 0.549 | 0.168 | 0.300 | 0.802 | 15.79 | 63.45 |
| `morpheme_bpe` | 65536 | 2.907 | 1.801 | 0.561 | 0.168 | 0.407 | 0.793 | 12.61 | 54.09 |
| `wp_bpe_ppig` | 65536 | 2.888 | 1.808 | 0.550 | 0.167 | 0.298 | 0.811 | 15.52 | 68.04 |
| `bpe_frequency` | 65536 | 2.944 | 1.765 | 0.564 | 0.166 | 0.280 | 0.807 | 13.33 | 57.37 |
| `length_penalized` | 65536 | 2.951 | 1.759 | 0.566 | 0.166 | 0.279 | 0.808 | 15.52 | 68.68 |
| `wp_bpe_npmi` | 65536 | 2.756 | 1.892 | 0.531 | 0.166 | 0.318 | 0.788 | 16.79 | 65.50 |
| `unigram_lm` | 65536 | 2.828 | 1.836 | 0.549 | 0.166 | 0.338 | 0.797 | 7.09 | 57.45 |
| `mdl_bits` | 40496 | 2.509 | 1.928 | 0.521 | 0.152 | 0.318 | 0.796 | 17.61 | 62.33 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.314 | 0.162 | 0.509 | 0.152 | 0.253 | 0.385 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.193 | 0.344 | 0.126 | 0.196 | 0.268 |
| `wordpiece_pmi` | 0.212 | 0.193 | 0.344 | 0.126 | 0.196 | 0.268 |
| `wp_bpe_scp` | 0.367 | 0.226 | 0.496 | 0.173 | 0.284 | 0.403 |
| `wp_bpe_pig` | 0.350 | 0.190 | 0.537 | 0.155 | 0.265 | 0.413 |
| `morpheme_bpe` | 0.465 | 0.306 | 0.534 | 0.370 | 0.358 | 0.476 |
| `wp_bpe_ppig` | 0.348 | 0.184 | 0.540 | 0.160 | 0.258 | 0.413 |
| `bpe_frequency` | 0.315 | 0.164 | 0.511 | 0.152 | 0.255 | 0.387 |
| `length_penalized` | 0.311 | 0.163 | 0.507 | 0.158 | 0.253 | 0.384 |
| `wp_bpe_npmi` | 0.388 | 0.219 | 0.532 | 0.164 | 0.286 | 0.426 |
| `unigram_lm` | 0.411 | 0.246 | 0.539 | 0.241 | 0.253 | 0.441 |
| `mdl_bits` | 0.400 | 0.233 | 0.512 | 0.163 | 0.284 | 0.421 |
<!-- END GENERATED: intrinsic_v65536 -->

</details>
