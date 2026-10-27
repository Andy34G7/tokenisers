# Extrinsic Proxy-LM Validation Bits-Per-Byte (BPB) Leaderboard

- **Compute Backend**: `CPU` (`cpu`)
- **Fixed UTF-8 Training Budget**: `150,000` bytes per tokenizer

| Tokenizer Variant | Train Bytes/Token | Tokens Trained (FLOPs proxy) | English BPB (↓) | Multilingual BPB (↓) | Code BPB (↓) | Macro Val BPB (↓) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 1.840 | 81,920 | 1.938 | 2.277 | 3.502 | **2.572** |
| `unigram_lm` | 1.808 | 82,944 | 2.204 | 2.268 | 3.388 | **2.620** |
| `bpe_frequency` | 1.674 | 90,112 | 2.207 | 2.590 | 3.417 | **2.738** |
| `bpe_dropout_p0.1` | 1.673 | 90,112 | 2.228 | 2.615 | 3.516 | **2.786** |
| `morpheme_bpe` | 1.622 | 92,672 | 2.331 | 2.300 | 3.749 | **2.793** |
| `length_penalized` | 1.649 | 91,136 | 2.197 | 2.602 | 3.614 | **2.804** |
| `mdl_bits` | 1.506 | 99,840 | 2.980 | 2.435 | 3.962 | **3.126** |
| `wordpiece_pmi` | 1.066 | 140,800 | 3.791 | 2.680 | 4.231 | **3.567** |
