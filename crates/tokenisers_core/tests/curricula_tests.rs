use tokenisers_core::curricula::morpheme::{MorphemeGuidedScorer, MorphemeLexicon};
use tokenisers_core::curricula::super_bpe::{SuperBpeConfig, SuperBpeTrainer};
use tokenisers_core::index::Pair;
use tokenisers_core::scoring::{BpeScorer, PairContext, PairScorer};

#[test]
fn test_superbpe_config_validation() {
    assert!(SuperBpeConfig::new(250, 300).is_err()); // v_switch < 256
    assert!(SuperBpeConfig::new(350, 300).is_err()); // v_switch > v_final
    assert!(SuperBpeConfig::new(280, 320).is_ok());
}

#[test]
fn test_superbpe_text_segmentation() {
    let text = "Hello   world!\nThis is a test.";
    let segments = SuperBpeTrainer::segment_text(text);
    let recovered: Vec<u8> = segments.into_iter().flatten().collect();
    assert_eq!(recovered, text.as_bytes());
}

#[test]
fn segment_text_splits_on_unicode_whitespace() {
    let segs = |t: &str| -> Vec<String> {
        SuperBpeTrainer::segment_text(t)
            .into_iter()
            .map(|s| String::from_utf8(s).unwrap())
            .collect()
    };
    assert_eq!(segs("中文\u{3000}文本"), ["中文", "\u{3000}", "文本"]);
    assert_eq!(segs("a\u{a0}b \u{2003}\tc"), ["a", "\u{a0}", "b", " \u{2003}\t", "c"]);
    assert_eq!(segs("x\u{2028}\u{85}y"), ["x", "\u{2028}\u{85}", "y"]);
    // Not White_Space: zero-width space, BOM, Mongolian vowel separator.
    assert_eq!(segs("a\u{200b}\u{feff}\u{180e}b"), ["a\u{200b}\u{feff}\u{180e}b"]);
    assert_eq!(segs("  lead"), ["  ", "lead"]);
    assert!(segs("").is_empty());
}

#[test]
fn segment_bytes_matches_segment_text_and_tolerates_invalid_utf8() {
    let text = "Hello\u{3000}世界 \t\n🚀x";
    let from_bytes: Vec<Vec<u8>> = SuperBpeTrainer::segment_bytes(text.as_bytes())
        .into_iter()
        .map(<[u8]>::to_vec)
        .collect();
    assert_eq!(from_bytes, SuperBpeTrainer::segment_text(text));

    // Invalid bytes are non-whitespace; segments always concatenate back to the input.
    let raw = b"a\xff \xc3\xe3\x80\x80 \x80";
    let segs = SuperBpeTrainer::segment_bytes(raw);
    assert_eq!(segs, vec![&b"a\xff"[..], b" ", b"\xc3", b"\xe3\x80\x80 ", b"\x80"]);
    assert_eq!(segs.concat(), raw.to_vec());
}

#[test]
fn super_bpe_marks_whole_text_encoding_only_after_stage_two() {
    let lines = vec!["the quick brown fox jumps over the lazy dog"; 4];
    let scorer = BpeScorer::new();
    let stage1_only = SuperBpeTrainer::new(SuperBpeConfig::new(270, 270).unwrap(), &scorer)
        .train(&lines)
        .unwrap();
    assert!(stage1_only.pretokenize);
    let two_stage = SuperBpeTrainer::new(SuperBpeConfig::new(270, 285).unwrap(), &scorer)
        .train(&lines)
        .unwrap();
    assert!(!two_stage.pretokenize);
}

#[test]
fn test_superbpe_two_stage_training() {
    let lines = vec![
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog",
    ];

    let config = SuperBpeConfig::new(270, 285).unwrap();
    let scorer = BpeScorer::new();
    let trainer = SuperBpeTrainer::new(config, &scorer);

    let model = trainer.train(&lines).expect("SuperBPE training failed");
    assert!(model.vocab.len() >= 270);
    assert!(model.vocab.len() <= 290);

    // Verify lossless round-trip
    let test_str = "the quick brown fox";
    let tokens = model.encode(test_str.as_bytes());
    let decoded = model.decode_lossless(&tokens).expect("SuperBPE decode failed");
    assert_eq!(test_str, decoded);
}

#[test]
fn test_morpheme_prior_amplification_and_attenuation() {
    let mut lexicon = MorphemeLexicon::new();
    // "unhappiness" -> "un" | "happi" | "ness" (byte offsets 2 and 7)
    lexicon.insert_segmentation(b"unhappiness", &[b"un", b"happi", b"ness"]);

    let bpe = BpeScorer::new();
    let mut scorer = MorphemeGuidedScorer::new(bpe, 2.0, 1.5);

    let pair_intra = Pair::new(b'h' as u32, b'a' as u32);
    let pair_cross = Pair::new(b'n' as u32, b'h' as u32);
    let pair_neutral = Pair::new(b'x' as u32, b'y' as u32);

    // Record 'h'-'a' at offset 3 (inside "happi" -> intra-morphemic)
    scorer.record_occurrence(pair_intra, b"unhappiness", 3, &lexicon);
    // Record 'n'-'h' at offset 2 (boundary between "un" and "happi" -> cross-boundary)
    scorer.record_occurrence(pair_cross, b"unhappiness", 2, &lexicon);

    assert_eq!(scorer.concordance(pair_intra), 1.0);
    assert_eq!(scorer.concordance(pair_cross), -1.0);
    assert_eq!(scorer.concordance(pair_neutral), 0.0);

    let ctx = PairContext {
        freq_uv: 100,
        freq_u: 100,
        freq_v: 100,
        total_tokens: 1000,
        active_vocab_size: 256,
        len_u: 1,
        len_v: 1,
    };

    let base_score = 100.0;
    let score_intra = scorer.score(pair_intra, &ctx);
    let score_cross = scorer.score(pair_cross, &ctx);
    let score_neutral = scorer.score(pair_neutral, &ctx);

    // Intra-morphemic should be strictly boosted: base * exp(2.0 * 1.0) = base * e^2
    assert!((score_intra - (base_score * 2.0f64.exp())).abs() < 1e-4);
    // Cross-boundary should be strictly penalized: base * exp(2.0 * -1.0) = base * e^-2
    assert!((score_cross - (base_score * (-2.0f64).exp())).abs() < 1e-4);
    // Neutral should remain unaffected: base * exp(0) = base
    assert!((score_neutral - base_score).abs() < 1e-4);
}

#[test]
fn test_morpheme_negative_base_score_prior_ranking() {
    let mut lexicon = MorphemeLexicon::new();
    lexicon.insert_segmentation(b"unhappiness", &[b"un", b"happi", b"ness"]);

    // Dummy negative scorer simulating MDL compression loss
    #[derive(Debug, Clone)]
    struct NegativeScorer(f64);
    impl PairScorer for NegativeScorer {
        fn score(&self, _pair: Pair, _ctx: &PairContext) -> f64 {
            self.0
        }
        fn name(&self) -> &'static str {
            "negative_scorer"
        }
    }

    let inner = NegativeScorer(-10.0);
    let mut scorer = MorphemeGuidedScorer::new(inner, 1.5, 1.0);

    let pair_intra = Pair::new(b'h' as u32, b'a' as u32);
    let pair_cross = Pair::new(b'n' as u32, b'h' as u32);

    scorer.record_occurrence(pair_intra, b"unhappiness", 3, &lexicon);
    scorer.record_occurrence(pair_cross, b"unhappiness", 2, &lexicon);

    let ctx = PairContext {
        freq_uv: 10,
        freq_u: 10,
        freq_v: 10,
        total_tokens: 100,
        active_vocab_size: 256,
        len_u: 1,
        len_v: 1,
    };

    let score_intra = scorer.score(pair_intra, &ctx);
    let score_cross = scorer.score(pair_cross, &ctx);

    // For negative base scores, intra-morphemic (M=1.0) must be higher (less negative) than cross-boundary (M=-1.0)
    assert!(
        score_intra > score_cross,
        "Intra-morphemic score {score_intra} should be higher than cross-boundary score {score_cross}"
    );
    // Expected: -10.0 / exp(1.5) ≈ -2.23 > -10.0 / exp(-1.5) ≈ -44.82
    assert!((score_intra - (-10.0 / 1.5f64.exp())).abs() < 1e-4);
    assert!((score_cross - (-10.0 / (-1.5f64).exp())).abs() < 1e-4);
}

