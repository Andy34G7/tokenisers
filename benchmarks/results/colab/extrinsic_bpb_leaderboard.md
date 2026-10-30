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
