# Extrinsic Proxy-LM Validation Bits-Per-Byte (BPB) Leaderboard

- **Compute Backend**: `CPU` (`cpu`)
- **Fixed UTF-8 Training Budget**: `150,000` bytes per tokenizer

| Tokenizer Variant | Train Bytes/Token | Tokens Trained (FLOPs proxy) | English BPB (↓) | Multilingual BPB (↓) | Code BPB (↓) | Macro Val BPB (↓) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 1.840 | 81,920 | 1.938 | 2.277 | 3.502 | **2.572** |
| `bpe_frequency` | 1.674 | 90,112 | 2.207 | 2.590 | 3.417 | **2.738** |
| `bpe_dropout_p0.1` | 1.673 | 90,112 | 2.228 | 2.615 | 3.516 | **2.786** |
| `morpheme_bpe` | 1.622 | 92,672 | 2.331 | 2.300 | 3.749 | **2.793** |
| `unigram_lm` | 1.801 | 83,456 | 2.378 | 2.672 | 3.331 | **2.794** |
| `length_penalized` | 1.649 | 91,136 | 2.197 | 2.602 | 3.614 | **2.804** |
| `wp_bpe_pig` | 1.588 | 94,720 | 2.659 | 2.212 | 3.565 | **2.812** |
| `wp_bpe_ppig` | 1.594 | 94,208 | 2.355 | 2.614 | 3.528 | **2.832** |
| `wp_bpe_npmi` | 1.497 | 100,352 | 2.767 | 2.315 | 3.838 | **2.973** |
| `mdl_bits` | 1.506 | 99,840 | 2.980 | 2.435 | 3.962 | **3.126** |
| `wp_bpe_scp` | 1.288 | 116,736 | 3.533 | 2.252 | 3.871 | **3.219** |
| `wordpiece_pmi` | 1.066 | 140,800 | 3.791 | 2.680 | 4.231 | **3.567** |
| `wp_bpe_ft` | 1.066 | 140,800 | 3.791 | 2.680 | 4.231 | **3.567** |
| `wp_bpe_base` | 1.066 | 140,800 | 3.911 | 2.715 | 4.276 | **3.634** |
