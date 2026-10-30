mod common;

use common::{random_corpus, reference_train};
use tokenisers_core::curricula::morpheme::{MorphemeGuidedScorer, MorphemeLexicon, PairMorphologyStats};
use tokenisers_core::curricula::super_bpe::{SuperBpeConfig, SuperBpeTrainer};
use tokenisers_core::engine::bottom_up::{BottomUpModel, BottomUpTrainer};
use tokenisers_core::index::Pair;
use tokenisers_core::scoring::{
    BpeScorer, LengthPenalizedScorer, MdlScorer, PairContext, PairScorer, WordPieceScorer, WpBpePigScorer,
};

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

// =========================================================================
// Morpheme-guided BPE
// =========================================================================

fn lexicon(entries: &[(&str, &[&str])]) -> MorphemeLexicon {
    let mut lex = MorphemeLexicon::new();
    for (word, morphemes) in entries {
        let parts: Vec<&[u8]> = morphemes.iter().map(|m| m.as_bytes()).collect();
        assert!(lex.insert_segmentation(word.as_bytes(), &parts), "{word}");
    }
    lex
}

fn pieces(model: &BottomUpModel, word: &str) -> Vec<String> {
    model
        .encode(word.as_bytes())
        .iter()
        .map(|&id| String::from_utf8_lossy(model.vocab.get_bytes(id).unwrap()).into_owned())
        .collect()
}

fn ctx(freq_uv: u64) -> PairContext {
    PairContext {
        freq_uv,
        merges_uv: freq_uv,
        freq_u: 100,
        freq_v: 100,
        total_tokens: 1000,
        active_vocab_size: 256,
        len_u: 1,
        len_v: 1,
    }
}

#[test]
fn morpheme_lexicon_stores_inner_byte_offsets_and_rejects_non_surface_segmentations() {
    let lex = lexicon(&[("unhappiness", &["un", "happi", "ness"]), ("перестройка", &["пере", "строй", "ка"])]);
    assert_eq!(lex.get_boundaries(b"unhappiness"), Some(&[2usize, 7][..]));
    assert_eq!(lex.get_boundaries("перестройка".as_bytes()), Some(&[8usize, 18][..]));
    assert_eq!(lex.get_boundaries(b"happiness"), None);

    let mut lex = MorphemeLexicon::new();
    // Empty morphemes add no boundary; a single morpheme gives an annotated word without one.
    assert!(lex.insert_segmentation(b"ab", &[b"", b"a", b"", b"b", b""]));
    assert!(lex.insert_segmentation(b"cat", &[b"cat"]));
    assert_eq!(lex.get_boundaries(b"ab"), Some(&[1usize][..]));
    assert_eq!(lex.get_boundaries(b"cat"), Some(&[][..]));
    // Canonical (non-surface) segmentations would give wrong offsets: rejected, not inserted.
    assert!(!lex.insert_segmentation(b"happiness", &[b"happy", b"ness"]));
    assert!(!lex.insert_segmentation(b"short", &[b"sho", b"rt", b"extra"]));
    assert_eq!(lex.get_boundaries(b"happiness"), None);
    assert_eq!(lex.len(), 2);
}

#[test]
fn morpheme_scorer_scales_by_concordance_and_vetoes_sub_unit_counts() {
    let pair = Pair::new(b'h' as u32, b'a' as u32);
    let intra = PairMorphologyStats { n_intra: 10, n_cross: 0 };
    let cross = PairMorphologyStats { n_intra: 0, n_cross: 10 };
    let mixed = PairMorphologyStats { n_intra: 3, n_cross: 1 };
    let scorer = MorphemeGuidedScorer::new(BpeScorer::new(), MorphemeLexicon::new(), 2.0, 1.0);

    assert_eq!(scorer.concordance(intra), 1.0);
    assert_eq!(scorer.concordance(cross), -1.0);
    assert_eq!(scorer.concordance(mixed), 0.5);
    assert_eq!(scorer.concordance(PairMorphologyStats::default()), 0.0);

    let c = ctx(100);
    assert!((scorer.score_morph(pair, &c, intra) - 100.0 * 2.0f64.exp()).abs() < 1e-9);
    assert!((scorer.score_morph(pair, &c, cross) - 100.0 * (-2.0f64).exp()).abs() < 1e-9);
    assert_eq!(scorer.score_morph(pair, &c, PairMorphologyStats::default()), 100.0);
    assert_eq!(scorer.score(pair, &c), 100.0);
    // 5 * e^-2 < 1 occurrence: not a merge candidate.
    assert_eq!(scorer.score_morph(pair, &ctx(5), cross), f64::NEG_INFINITY);
    assert!(scorer.score_morph(pair, &ctx(8), cross).is_finite());

    // Negative base scores (MDL-like): intra less negative than cross.
    #[derive(Debug)]
    struct Negative;
    impl PairScorer for Negative {
        fn score(&self, _: Pair, _: &PairContext) -> f64 {
            -10.0
        }
        fn name(&self) -> &'static str {
            "negative"
        }
    }
    let neg = MorphemeGuidedScorer::new(Negative, MorphemeLexicon::new(), 1.5, 1.0);
    assert!((neg.score_morph(pair, &c, intra) - (-10.0 / 1.5f64.exp())).abs() < 1e-9);
    assert!((neg.score_morph(pair, &c, cross) - (-10.0 * 1.5f64.exp())).abs() < 1e-9);

    // gamma weighs cross occurrences: (3 - 2 * 1) / 4.
    let gamma2 = MorphemeGuidedScorer::new(BpeScorer::new(), MorphemeLexicon::new(), 1.0, 2.0);
    assert_eq!(gamma2.concordance(mixed), 0.25);

    // Absurd lambdas stay finite (no inf/NaN that would silently drop intra pairs).
    for lambda in [1e6, f64::INFINITY] {
        let huge = MorphemeGuidedScorer::new(BpeScorer::new(), MorphemeLexicon::new(), lambda, 1.0);
        assert_eq!(huge.score_morph(pair, &c, intra), f64::MAX);
        assert_eq!(huge.score_morph(pair, &c, cross), f64::NEG_INFINITY);
        let neg = MorphemeGuidedScorer::new(Negative, MorphemeLexicon::new(), lambda, 1.0);
        let s = neg.score_morph(pair, &c, intra);
        assert!(s.is_finite() && s <= 0.0);
    }
}

/// Acceptance: a strong prior keeps every learned token inside the morphemes of an annotated word,
/// including merges of already-merged tokens (un + happy), while plain BPE crosses the boundary.
#[test]
fn strong_prior_never_learns_a_token_across_a_lexicon_boundary() {
    let corpus: Vec<(Vec<u8>, u64)> = [("unhappy", 100), ("happy", 40), ("unhappiness", 20), ("undo", 30)]
        .iter()
        .map(|&(w, f)| (w.as_bytes().to_vec(), f))
        .collect();
    let target = 256 + 40; // more merges than intra-morpheme pairs exist

    let bpe = BottomUpTrainer::new(&BpeScorer, target).train(&corpus).unwrap();
    assert_eq!(pieces(&bpe, "unhappy"), ["unhappy"], "plain BPE must cross un|happy for this test to mean anything");

    let lex = lexicon(&[("unhappy", &["un", "happy"])]);
    let guided = MorphemeGuidedScorer::new(BpeScorer, lex.clone(), 50.0, 1.0);
    let model = BottomUpTrainer::new(&guided, target).train(&corpus).unwrap();
    assert_eq!(pieces(&model, "unhappy"), ["un", "happy"]);
    // No learned token, wherever it occurs inside "unhappy", may span offset 2.
    for id in 256..model.vocab.len() as u32 {
        let tok = model.vocab.get_bytes(id).unwrap();
        for (p, window) in b"unhappy".windows(tok.len()).enumerate() {
            let crosses = p < 2 && p + tok.len() > 2;
            assert!(!(crosses && window == tok), "token {:?} crosses un|happy", String::from_utf8_lossy(tok));
        }
    }
    // Training stops rather than merge vetoed pairs; it does not pad.
    assert!(model.vocab.len() < target);

    // The same lexicon at lambda = 0, or an empty lexicon, is plain BPE.
    for scorer in [
        MorphemeGuidedScorer::new(BpeScorer, lex, 0.0, 1.0),
        MorphemeGuidedScorer::new(BpeScorer, MorphemeLexicon::new(), 50.0, 1.0),
    ] {
        let same = BottomUpTrainer::new(&scorer, target).train(&corpus).unwrap();
        assert_eq!(same.merges, bpe.merges);
    }
}

fn assert_matches_oracle<S: PairScorer>(
    name: &str,
    inner: S,
    lexicon: &MorphemeLexicon,
    corpus: &[(Vec<u8>, u64)],
    lambda: f64,
    gamma: f64,
) -> Vec<(u32, u32, u32)> {
    let scorer = MorphemeGuidedScorer::new(inner, lexicon.clone(), lambda, gamma);
    let target = 256 + 30;
    let model = BottomUpTrainer::new(&scorer, target).train(corpus).unwrap();
    let (ref_merges, ref_words) = reference_train(&scorer, corpus, target, 1);
    assert_eq!(model.merges, ref_merges, "{name}: merges differ from brute force");
    for ((word, _), seg) in corpus.iter().zip(&ref_words) {
        assert_eq!(&model.encode(word), seg, "{name}: encode != training segmentation");
    }
    model.merges
}

/// The engine's incremental intra/cross counts (every pair, merged tokens included) must equal a
/// from-scratch recount over the byte spans of the current segmentation at every step.
#[test]
fn morpheme_guided_training_matches_bruteforce_oracle() {
    let mut prior_changed_merges = 0;
    for seed in 0..120u64 {
        let mut rng = fastrand::Rng::with_seed(seed);
        let alphabet: &[u8] = if seed % 2 == 0 { b"ab" } else { b"abc" };
        let corpus = random_corpus(&mut rng, alphabet, 30, 10);
        let mut lex = MorphemeLexicon::new();
        for (word, _) in &corpus {
            if word.len() > 1 && rng.bool() {
                let mut cuts: Vec<usize> = (1..word.len()).filter(|_| rng.u8(0..3) == 0).collect();
                cuts.insert(0, 0);
                cuts.push(word.len());
                let parts: Vec<&[u8]> = cuts.windows(2).map(|c| &word[c[0]..c[1]]).collect();
                assert!(lex.insert_segmentation(word, &parts));
            }
        }
        for &(lambda, gamma) in &[(0.7, 1.0), (3.0, 2.0), (50.0, 1.0)] {
            let tag = format!("seed {seed} lambda {lambda} gamma {gamma}");
            let bpe = assert_matches_oracle(&format!("bpe {tag}"), BpeScorer, &lex, &corpus, lambda, gamma);
            let plain = BottomUpTrainer::new(&BpeScorer, 256 + 30).train(&corpus).unwrap();
            prior_changed_merges += usize::from(bpe != plain.merges);
            assert_matches_oracle(&format!("wordpiece {tag}"), WordPieceScorer::new(1.0, 1), &lex, &corpus, lambda, gamma);
            assert_matches_oracle(&format!("mdl {tag}"), MdlScorer::new(4.0, 1), &lex, &corpus, lambda, gamma);
            assert_matches_oracle(&format!("lp {tag}"), LengthPenalizedScorer::new(0.5, 1), &lex, &corpus, lambda, gamma);
            assert_matches_oracle(&format!("pig {tag}"), WpBpePigScorer::new(1), &lex, &corpus, lambda, gamma);
        }
    }
    assert!(prior_changed_merges > 100, "the prior must actually change training ({prior_changed_merges})");
}

#[test]
fn empty_or_unmatched_lexicon_trains_exactly_like_the_inner_scorer() {
    let text = common::pseudo_text(3, 30_000);
    let mut counts = std::collections::BTreeMap::new();
    for seg in SuperBpeTrainer::segment_text(&text) {
        *counts.entry(seg).or_insert(0u64) += 1;
    }
    let corpus: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
    let unmatched = lexicon(&[("zzzz", &["zz", "zz"])]);
    fn check<S: PairScorer + Clone>(inner: S, lexicons: &[&MorphemeLexicon], corpus: &[(Vec<u8>, u64)]) {
        let plain = BottomUpTrainer::new(&inner, 600).train(corpus).unwrap();
        for lex in lexicons {
            let guided = MorphemeGuidedScorer::new(inner.clone(), (*lex).clone(), 50.0, 1.0);
            let model = BottomUpTrainer::new(&guided, 600).train(corpus).unwrap();
            assert_eq!(model.merges, plain.merges, "{}", inner.name());
        }
    }
    let empty = MorphemeLexicon::new();
    let lexicons = [&empty, &unmatched];
    check(BpeScorer, &lexicons, &corpus);
    check(WordPieceScorer::new(1.0, 1), &lexicons, &corpus);
    check(MdlScorer::new(24.0, 1), &lexicons, &corpus);
    check(LengthPenalizedScorer::new(0.5, 1), &lexicons, &corpus);
}
