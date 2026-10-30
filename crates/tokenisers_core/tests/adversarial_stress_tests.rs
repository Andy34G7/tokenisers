use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use tokenisers_core::arena::{SymbolArena, NULL_NODE};
use tokenisers_core::curricula::morpheme::{MorphemeGuidedScorer, MorphemeLexicon};
use tokenisers_core::curricula::super_bpe::{SuperBpeConfig, SuperBpeTrainer};
use tokenisers_core::engine::bottom_up::{BottomUpTrainer, parallel_count_words};
use tokenisers_core::engine::unigram::{UnigramTrainer, logsumexp2};
use tokenisers_core::export::huggingface::HfTokenizerJson;
use tokenisers_core::index::Pair;
use tokenisers_core::inference::byte_entropy::{ByteEntropyConfig, ByteEntropyPatcher};
use tokenisers_core::inference::unigram_sample::UnigramSampler;
use tokenisers_core::scoring::frequency::BpeScorer;
use tokenisers_core::scoring::length_penalized::LengthPenalizedScorer;
use tokenisers_core::scoring::mdl::MdlScorer;
use tokenisers_core::scoring::pmi::WordPieceScorer;
use tokenisers_core::scoring::{PairContext, PairScorer};
use tokenisers_core::vocab::{TokenId, Vocabulary};

// =========================================================================
// 1. Pathological Inputs: Splicing & Pointer Invariants
// =========================================================================

#[test]
fn test_pathological_arena_splicing_invariants() {
    let mut arena = SymbolArena::new();

    // 1. Single character word
    let word_single = arena.add_word(&[42], 10);
    assert_eq!(arena.get_word_tokens(word_single), vec![42]);

    // 2. Empty word
    let word_empty = arena.add_word(&[], 5);
    assert_eq!(arena.get_word_tokens(word_empty), Vec::<TokenId>::new());

    // 3. Repeated pattern: 1000 identical tokens
    let tokens_1000: Vec<TokenId> = vec![65; 1000]; // 'A'
    let word_1000 = arena.add_word(&tokens_1000, 1);
    let mut word = *arena.get_word(word_1000).unwrap();
    assert_eq!(word.len, 1000);

    // Iteratively splice pairs from head to tail: (0, 1), (2, 3), etc.
    let mut curr = word.head;
    let mut spliced_count = 0;
    while curr != NULL_NODE {
        let node = *arena.get_node(curr).unwrap();
        let next = node.next;
        if next != NULL_NODE {
            let res = arena.splice_pair(curr, next, 999, word_1000);
            assert!(res.is_ok(), "Splicing must succeed on valid adjacent nodes");
            spliced_count += 1;
            let (_, next_next) = res.unwrap();
            curr = next_next;
        } else {
            break;
        }
    }
    assert_eq!(spliced_count, 500);
    word = *arena.get_word(word_1000).unwrap();
    assert_eq!(word.len, 500);

    let active_tokens = arena.get_word_tokens(word_1000);
    assert_eq!(active_tokens.len(), 500);
    for &tok in &active_tokens {
        assert_eq!(tok, 999);
    }
}

#[test]
fn test_pathological_inputs_bottom_up_training_and_encoding() {
    let pathological_corpus = vec![
        ("aaaa".repeat(250).into_bytes(), 10u64),      // 1000 'a's
        ("ab".repeat(2500).into_bytes(), 10u64),        // 5000 alternating 'ab's
        ("x".to_string().into_bytes(), 5u64),             // single character
        ("".to_string().into_bytes(), 20u64),           // empty word
        ("zzz".repeat(100).into_bytes(), 5u64),         // 300 'z's
    ];

    let scorer = BpeScorer;
    let trainer = BottomUpTrainer::new(&scorer, 300);
    let model = trainer.train(&pathological_corpus).expect("Training on pathological corpus must succeed");

    // Training stops early only when no pair is left: then every word is a single token.
    if model.vocab.len() < 300 {
        for (word, _) in &pathological_corpus {
            if !word.is_empty() {
                assert_eq!(model.encode(word).len(), 1, "stopped at {} with mergeable pairs left", model.vocab.len());
            }
        }
    }

    // Verify lossless encoding on repeated, alternating, single, and empty inputs
    let rep_a = "aaaa".repeat(250).into_bytes();
    let rep_ab = "ab".repeat(2500).into_bytes();
    let test_inputs: Vec<&[u8]> = vec![
        b"",
        b"a",
        b"aaaa",
        &rep_a,
        &rep_ab,
        b"x",
        b"zzzzzz",
    ];

    for input in test_inputs {
        let encoded = model.encode(input);
        let decoded = model.decode(&encoded).unwrap();
        assert_eq!(&decoded[..], input, "Lossless roundtrip failed for input of length {}", input.len());
    }
}

// =========================================================================
// 2. Extreme Byte Losslessness & Invalid UTF-8
// =========================================================================

#[test]
fn test_extreme_byte_losslessness_and_invalid_utf8() {
    // Construct rich corpus for training
    let corpus_text = [
        "Hello, World!",
        "Complex UTF-8: 𝄞 (U+1D11E) and 𠀀 (U+20000) and \u{10FFFF}",
        "Family Emoji: 👨‍👩‍👧‍👦 Flags: 🇺🇸🇩🇪🇯🇵 Skin: 👍🏽👩🏾‍💻",
        "Control & Whitespace: \r\n\t \x00\x01\x1f\x7f",
    ];
    let word_counts = parallel_count_words(&corpus_text);
    let counts_vec: Vec<(Vec<u8>, u64)> = word_counts.into_iter().collect();

    // Train BPE and Unigram
    let bpe_scorer = BpeScorer;
    let bpe_trainer = BottomUpTrainer::new(&bpe_scorer, 300);
    let bpe_model = bpe_trainer.train(&counts_vec).expect("BPE training failed");

    let unigram_trainer = UnigramTrainer::new(300, 0.2);
    let unigram_model = unigram_trainer.train(&counts_vec).expect("Unigram training failed");

    // Extreme adversarial byte sequences
    let adversarial_sequences: Vec<Vec<u8>> = vec![
        // 1. Bulk Null bytes
        vec![0u8; 500],
        // 2. Bulk High bytes
        vec![0xff; 500],
        // 3. Stray continuation byte (invalid UTF-8)
        vec![0x80],
        vec![0xBF],
        // 4. Overlong ASCII sequences (invalid UTF-8)
        vec![0xC0, 0xAF],
        vec![0xE0, 0x80, 0xAF],
        vec![0xF0, 0x80, 0x80, 0xAF],
        // 5. Incomplete sequences (truncated UTF-8)
        vec![0xC3],
        vec![0xE2, 0x82],
        vec![0xF0, 0x9F, 0x98],
        // 6. UTF-16 surrogates (invalid in UTF-8)
        vec![0xED, 0xA0, 0x80], // High surrogate D800
        vec![0xED, 0xBF, 0xBF], // Low surrogate DFFF
        // 7. Out of Unicode range (byte >= 0xF5)
        vec![0xF5, 0x80, 0x80, 0x80],
        vec![0xFF, 0xFE, 0xFD, 0xFC],
        // 8. High Unicode Planes
        "𝄞".as_bytes().to_vec(),           // Plane 1 U+1D11E
        "𠀀".as_bytes().to_vec(),           // Plane 2 U+20000
        "\u{10FFFF}".as_bytes().to_vec(),   // Plane 16 U+10FFFF
        // 9. Complex Emoji ZWJ combinations
        "👨‍👩‍👧‍👦".as_bytes().to_vec(),
        "🇺🇸🇩🇪🇯🇵".as_bytes().to_vec(),
        "👍🏽👩🏾‍💻".as_bytes().to_vec(),
        // 10. Mixed Whitespace combinations
        b"\r\n\t   \r\r\n\t\t\t   \x0B\x0C".to_vec(),
    ];

    for seq in &adversarial_sequences {
        // BPE Lossless Roundtrip
        let bpe_encoded = bpe_model.encode(seq);
        let bpe_decoded = bpe_model.decode(&bpe_encoded).unwrap();
        assert_eq!(&bpe_decoded, seq, "BPE failed raw byte losslessness on sequence {:02X?}", seq);

        // Unigram Lossless Roundtrip
        let uni_encoded = unigram_model.encode(seq);
        let uni_decoded = unigram_model.decode(&uni_encoded).unwrap();
        assert_eq!(&uni_decoded, seq, "Unigram failed raw byte losslessness on sequence {:02X?}", seq);

        // If valid UTF-8, test decode_lossless
        if let Ok(orig_str) = std::str::from_utf8(seq) {
            let bpe_str = bpe_model.decode_lossless(&bpe_encoded).unwrap();
            assert_eq!(bpe_str, orig_str, "BPE decode_lossless string mismatch");

            let uni_str = unigram_model.decode_lossless(&uni_encoded).unwrap();
            assert_eq!(uni_str, orig_str, "Unigram decode_lossless string mismatch");
        }
    }
}

#[test]
fn test_gpt2_byte_char_bijection_all_256_bytes() {
    let vocab = Vocabulary::new_byte_level();

    // Verify all 256 individual bytes
    for b in 0..=255u8 {
        let single = vec![b];
        let printable = vocab.bytes_to_printable(&single);
        let roundtrip = vocab.printable_to_bytes(&printable).expect("Bijective decoding must succeed");
        assert_eq!(roundtrip, single, "Bijective failure on single byte 0x{:02X}", b);
    }

    // Verify concatenated full 256-byte sequence
    let all_bytes: Vec<u8> = (0..=255u8).collect();
    let printable_all = vocab.bytes_to_printable(&all_bytes);
    let roundtrip_all = vocab.printable_to_bytes(&printable_all).expect("Full 256 byte decode must succeed");
    assert_eq!(roundtrip_all, all_bytes);
}

// =========================================================================
// 3. Numerical Stability of Unigram logsumexp2
// =========================================================================

#[test]
fn test_unigram_logsumexp2_extreme_floats() {
    let ln2 = 2.0f64.ln();

    // 1. Extreme negative values
    let val_neg_1000 = logsumexp2(-1000.0, -1000.0);
    assert!((val_neg_1000 - (-1000.0 + ln2)).abs() < 1e-10, "logsumexp2(-1000, -1000) failed");

    let val_neg_1e6 = logsumexp2(-1e6, -1e6);
    assert!((val_neg_1e6 - (-1e6 + ln2)).abs() < 1e-10, "logsumexp2(-1e6, -1e6) failed");

    let val_neg_1e10 = logsumexp2(-1e10, -1e10);
    assert!((val_neg_1e10 - (-1e10 + ln2)).abs() < 1e-10, "logsumexp2(-1e10, -1e10) failed");

    // 2. Extreme imbalance (|a - b| > 50.0)
    let val_imbalance_1 = logsumexp2(-1000.0, -1e6);
    assert_eq!(val_imbalance_1, -1000.0, "Large negative imbalance must return max");

    let val_imbalance_2 = logsumexp2(-1e6, -1000.0);
    assert_eq!(val_imbalance_2, -1000.0, "Large negative imbalance reverse must return max");

    let val_imbalance_3 = logsumexp2(0.0, -100.0);
    assert_eq!(val_imbalance_3, 0.0);

    // 3. Negative Infinity
    let neg_inf = f64::NEG_INFINITY;
    assert_eq!(logsumexp2(neg_inf, -500.0), -500.0);
    assert_eq!(logsumexp2(-500.0, neg_inf), -500.0);
    assert_eq!(logsumexp2(neg_inf, neg_inf), neg_inf);

    // 4. Extreme positive values
    let val_pos_1e6 = logsumexp2(1e6, 1e6);
    assert!((val_pos_1e6 - (1e6 + ln2)).abs() < 1e-10);

    // 5. Zero values
    let val_zeros = logsumexp2(0.0, 0.0);
    assert!((val_zeros - ln2).abs() < 1e-10);

    // Guarantee no NaNs produced across entire grid
    let test_points = [-1e12, -1e9, -1e6, -1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1e6, neg_inf];
    for &a in &test_points {
        for &b in &test_points {
            let res = logsumexp2(a, b);
            assert!(!res.is_nan(), "logsumexp2({}, {}) produced NaN!", a, b);
        }
    }
}

// =========================================================================
// 4. Pruning Invariants: Base 256 Byte Protection
// =========================================================================

#[test]
fn test_unigram_pruning_base_byte_invariants_minimal_vocab() {
    let corpus = vec![
        (b"supercalifragilisticexpialidocious".to_vec(), 100u64),
        (b"antidisestablishmentarianism".to_vec(), 100u64),
        (b"pseudopseudohypoparathyroidism".to_vec(), 100u64),
        (b"pneumonoultramicroscopicsilicovolcanoconiosis".to_vec(), 100u64),
    ];

    // Train with minimum possible vocabulary: target_vocab_size = 256
    let trainer = UnigramTrainer::new(256, 0.3);
    let model = trainer.train(&corpus).expect("Unigram training to minimal vocab 256 must succeed");

    assert_eq!(model.vocab.len(), 256, "Vocabulary must be pruned down to exactly 256 base bytes");

    // Verify all 256 base bytes 0..=255 exist and map to their respective byte
    for b in 0..=255u8 {
        let token_bytes = model.vocab.get_bytes(b as TokenId);
        assert_eq!(
            token_bytes,
            Some(&[b][..]),
            "Base byte 0x{:02X} was evicted or corrupted during minimal pruning!",
            b
        );
    }

    // Verify lossless fallback encoding on any input
    let test_bytes = b"testing arbitrary text with \x00 and \xff bytes";
    let encoded = model.encode(test_bytes);
    let decoded = model.decode(&encoded).unwrap();
    assert_eq!(&decoded, test_bytes, "Lossless fallback failed with minimal 256 vocabulary");
}

// =========================================================================
// 5. Dynamic Inference Boundaries: BPE-Dropout & Byte-Entropy
// =========================================================================

#[test]
fn test_bpe_dropout_boundary_conditions() {
    let corpus = vec![
        (b"internationalization".to_vec(), 50u64),
        (b"interconnection".to_vec(), 40u64),
        (b"interstellar".to_vec(), 30u64),
    ];
    let scorer = BpeScorer;
    let trainer = BottomUpTrainer::new(&scorer, 300);
    let model = trainer.train(&corpus).expect("Training failed");

    let input = b"internationalization and interconnection";

    // 1. p = 0.0: strictly deterministic greedy BPE
    let encoded_greedy = model.encode(input);
    let dropout_p0 = model.encode_with_dropout(input, 0.0, None);
    assert_eq!(
        dropout_p0, encoded_greedy,
        "BPE-Dropout with p=0.0 must be strictly identical to deterministic greedy BPE"
    );

    // 2. p = 1.0: maximum dropout fallback to raw base byte tokens
    let raw_bytes: Vec<TokenId> = input.iter().map(|&b| b as TokenId).collect();
    let dropout_p1 = model.encode_with_dropout(input, 1.0, None);
    assert_eq!(
        dropout_p1, raw_bytes,
        "BPE-Dropout with p=1.0 must fall back entirely to raw base byte tokens"
    );

    // 3. Out-of-bounds p < 0.0 and p > 1.0 handling
    let dropout_neg = model.encode_with_dropout(input, -0.5, None);
    assert_eq!(dropout_neg, encoded_greedy, "p < 0.0 must clamp to greedy p=0.0");

    let dropout_over = model.encode_with_dropout(input, 1.5, None);
    assert_eq!(dropout_over, raw_bytes, "p > 1.0 must clamp to raw bytes p=1.0");

    // 4. Lossless decoding across all dropout probabilities and random seeds
    for &p in &[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
        for seed in 0..10 {
            let encoded = model.encode_with_dropout(input, p, Some(seed));
            let decoded = model.decode(&encoded).unwrap();
            assert_eq!(&decoded, input, "BPE-Dropout losslessness violated at p={}, seed={}", p, seed);
        }
    }
}

#[test]
fn test_byte_entropy_extreme_thresholds() {
    let corpus = b"The quick brown fox jumps over the lazy dog. Programming in Rust is fast and safe.";
    let mut patcher = ByteEntropyPatcher::new(ByteEntropyConfig {
        context_order: 3,
        tau_threshold: 4.2,
        min_len: 2,
        max_len: 16,
    });
    patcher.train_transitions(corpus);

    let bulk_zeros = vec![0u8; 1000];
    let bulk_ff = vec![0xff; 1000];
    let test_inputs: Vec<&[u8]> = vec![
        b"",
        b"a",
        b"hello world",
        corpus,
        &bulk_zeros,
        &bulk_ff,
    ];

    for input in test_inputs {
        // Extreme 1: tau = 0.0 (maximum sensitivity -> splits eagerly at min_len)
        let mut patcher_tau0 = patcher.clone();
        patcher_tau0.config.tau_threshold = 0.0;
        let patches_tau0 = patcher_tau0.segment_stream(input);
        let recon_tau0: Vec<u8> = patches_tau0.into_iter().flatten().collect();
        assert_eq!(&recon_tau0, input, "Byte-entropy failed losslessness at tau=0.0");

        // Extreme 2: tau = 100.0 (minimum sensitivity -> splits only at max_len)
        let mut patcher_tau100 = patcher.clone();
        patcher_tau100.config.tau_threshold = 100.0;
        let patches_tau100 = patcher_tau100.segment_stream(input);
        let recon_tau100: Vec<u8> = patches_tau100.into_iter().flatten().collect();
        assert_eq!(&recon_tau100, input, "Byte-entropy failed losslessness at tau=100.0");
    }
}

// =========================================================================
// 6. SuperBPE Stage Transition & Infinite Loop Stress Test
// =========================================================================

#[test]
fn test_super_bpe_stage_transition_when_intra_word_pairs_exhaust() {
    // Stress test: what happens when intra-word pairs exhaust before reaching v_switch?
    // v_switch = 300, v_final = 350.
    // Corpus has only a short line: "short text", with at most ~8 intra-word pairs.
    // If the trainer has an infinite loop bug when intra-word pairs are exhausted,
    // this test will catch it via a thread timeout channel.
    let (tx, rx) = mpsc::channel();

    let handle = thread::spawn(move || {
        let config = match SuperBpeConfig::new(300, 350) {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(Err(format!("Config error: {:?}", e)));
                return;
            }
        };
        let scorer = BpeScorer;
        let trainer = SuperBpeTrainer::new(config, &scorer);
        let result = trainer.train(&["short text"]);
        let _ = tx.send(result.map(|m| m.vocab.len()).map_err(|e| format!("{:?}", e)));
    });

    let timeout = Duration::from_millis(2000); // 2 seconds timeout
    match rx.recv_timeout(timeout) {
        Ok(train_res) => {
            println!("SuperBpeTrainer finished within timeout: {:?}", train_res);
            assert!(
                train_res.is_ok(),
                "SuperBpeTrainer should complete successfully without error: {:?}",
                train_res
            );
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!(
                "CRITICAL BUG CONFIRMED: SuperBpeTrainer::train hung in an INFINITE LOOP! \
                Intra-word pairs exhausted before v_switch (300), causing infinite loop on vocab.add_token(vec![0x20])."
            );
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let _ = handle.join();
            panic!("SuperBpeTrainer worker thread crashed or disconnected unexpectedly!");
        }
    }
}

// =========================================================================
// 7. Unigram Sampler: FFBS & k-Best Boundary Tests
// =========================================================================

#[test]
fn test_unigram_sampler_boundary_temperatures_and_k() {
    let corpus = vec![
        (b"algorithmic".to_vec(), 50u64),
        (b"algorithm".to_vec(), 40u64),
        (b"algebra".to_vec(), 30u64),
    ];
    let trainer = UnigramTrainer::new(300, 0.2);
    let model = trainer.train(&corpus).expect("Unigram training failed");

    let sampler = UnigramSampler::new(&model.trie);
    let test_inputs: Vec<&[u8]> = vec![
        b"",
        b"a",
        b"algorithm",
        b"algebraic algorithm",
        b"\x00\xff invalid utf8 \x80\xc0\xaf",
    ];

    for input in test_inputs {
        // FFBS at extreme temperatures
        for &temp in &[1e-5, 0.1, 1.0, 10.0, 100.0] {
            let tokens = sampler.sample_ffbs(input, temp, Some(42));
            let decoded = model.decode(&tokens).unwrap();
            assert_eq!(&decoded, input, "FFBS losslessness failed at temp {}", temp);
        }

        // k-Best at k = 1, k = 5, k = 100
        for &k in &[0, 1, 5, 50] {
            let tokens = sampler.sample_kbest(input, k, 1.0, Some(42));
            let decoded = model.decode(&tokens).unwrap();
            assert_eq!(&decoded, input, "k-Best losslessness failed at k = {}", k);
        }
    }
}

// =========================================================================
// 8. Scorers: Boundary and Extreme Values
// =========================================================================

#[test]
fn test_scorers_extreme_and_boundary_values() {
    // 1. MDL Scorer with tiny counts and zero total tokens
    let mdl = MdlScorer::new(24.0, 2);
    let ctx_zero = PairContext {
        freq_uv: 1,
        freq_u: 1,
        freq_v: 1,
        total_tokens: 0,
        active_vocab_size: 256,
        len_u: 1,
        len_v: 1,
    };
    assert_eq!(mdl.score(Pair::new(0, 1), &ctx_zero), f64::NEG_INFINITY);

    // 2. WordPiece with zero count and epsilon stability
    let wp = WordPieceScorer::new(1.0, 5);
    let ctx_low_freq = PairContext {
        freq_uv: 3, // < min_frequency 5
        freq_u: 10,
        freq_v: 10,
        total_tokens: 100,
        active_vocab_size: 256,
        len_u: 1,
        len_v: 1,
    };
    assert_eq!(wp.score(Pair::new(0, 1), &ctx_low_freq), f64::NEG_INFINITY);

    // 3. Length-Penalized with extreme beta (+10.0 and -10.0)
    let lp_high = LengthPenalizedScorer::new(10.0, 1);
    let lp_low = LengthPenalizedScorer::new(-10.0, 1);
    let ctx_norm = PairContext {
        freq_uv: 10,
        freq_u: 20,
        freq_v: 20,
        total_tokens: 100,
        active_vocab_size: 256,
        len_u: 4,
        len_v: 4,
    };
    // count / len^beta: +10 penalizes the 8-byte merge, -10 rewards it; both stay finite.
    let score_high = lp_high.score(Pair::new(0, 1), &ctx_norm);
    let score_low = lp_low.score(Pair::new(0, 1), &ctx_norm);
    assert!(score_high.is_finite() && score_high > 0.0 && score_high < 10.0);
    assert!(score_low.is_finite() && score_low > 10.0);

    // 4. Morpheme-Guided with extreme lambda
    let bpe = BpeScorer;
    let mut morpheme_scorer = MorphemeGuidedScorer::new(bpe, 100.0, 2.0);
    let mut lexicon = MorphemeLexicon::new();
    lexicon.insert_segmentation(b"unhappy", &[b"un", b"happy"]);

    // Test cross-boundary penalty (extreme attenuation)
    morpheme_scorer.record_occurrence(Pair::new(10, 20), b"unhappy", 2, &lexicon);
    let score_cross = morpheme_scorer.score(Pair::new(10, 20), &ctx_norm);
    assert!(score_cross.is_finite() && score_cross >= 0.0);
}

// =========================================================================
// 9. HuggingFace Export: Pathological and Binary Models
// =========================================================================

#[test]
fn test_hf_export_pathological_and_binary_models() {
    let pathological_corpus = vec![
        (vec![0x00, 0x01, 0x00, 0x01], 10u64),
        (vec![0xff, 0xfe, 0xff, 0xfe], 10u64),
        (b"regular text".to_vec(), 10u64),
    ];
    let scorer = BpeScorer;
    let trainer = BottomUpTrainer::new(&scorer, 270);
    let model = trainer.train(&pathological_corpus).expect("BPE training failed");

    let hf_json = HfTokenizerJson::from_bpe_model(&model, None);
    let json_str = hf_json.to_json_string().expect("JSON serialization must succeed");

    assert!(json_str.contains("\"version\": \"1.0\""));
    assert!(json_str.contains("\"type\": \"ByteLevel\""));
    assert!(json_str.contains("\"merges\""));

    // Verify deserialization back to HfTokenizerJson
    let deserialized: HfTokenizerJson = serde_json::from_str(&json_str).expect("JSON deserialization must succeed");
    assert_eq!(deserialized.version, "1.0");
}
