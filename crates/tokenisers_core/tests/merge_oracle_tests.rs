//! The training engine must merge exactly what a brute-force recount + rescore would merge, for
//! every scorer, and encoding must reproduce the training segmentation.
mod common;

use common::{random_corpus, reference_encode, reference_train};
use tokenisers_core::curricula::morpheme::MorphemeGuidedScorer;
use tokenisers_core::engine::bottom_up::{BottomUpModel, BottomUpTrainer};
use tokenisers_core::index::Pair;
use tokenisers_core::scoring::{
    BpeScorer, LengthPenalizedScorer, MdlScorer, PairContext, PairScorer, ScoreDeps, WordPieceScorer,
    WpBpeBaseScorer, WpBpeFtScorer, WpBpeNpmiScorer, WpBpePigScorer, WpBpePpigScorer, WpBpeScpScorer,
};

/// Reads every context field in a non-monotone way and keeps the default (`Arbitrary`) deps.
#[derive(Debug)]
struct OddScorer;

impl PairScorer for OddScorer {
    fn score(&self, pair: Pair, ctx: &PairContext) -> f64 {
        let wobble = ((ctx.total_tokens + ctx.active_vocab_size as u64 + pair.0 as u64) % 5) as f64;
        ctx.freq_uv as f64 * (1.0 + wobble) - (ctx.freq_u as f64).sqrt()
    }
    fn name(&self) -> &'static str {
        "odd"
    }
}

/// Monotone in N up to a rounding-sized wobble (< 1e-12), like float noise in a real
/// `MonotoneTotals` scorer. The engine's tolerance band must absorb it.
#[derive(Debug)]
struct NoisyMonotoneScorer;

impl PairScorer for NoisyMonotoneScorer {
    fn score(&self, pair: Pair, ctx: &PairContext) -> f64 {
        let wobble = ((ctx.total_tokens * 7919 + pair.0 as u64 * 31 + pair.1 as u64) % 5) as f64;
        ctx.freq_uv as f64 + wobble * 1e-13
    }
    fn name(&self) -> &'static str {
        "noisy_monotone"
    }
    fn deps(&self) -> ScoreDeps {
        ScoreDeps::MonotoneTotals
    }
}

fn scorers() -> Vec<(&'static str, Box<dyn PairScorer>, u64)> {
    let mut morpheme = MorphemeGuidedScorer::new(WpBpePigScorer::new(1), 2.0, 1.0);
    morpheme.set_stats(Pair::new(b'a' as u32, b'b' as u32), 0, 5);
    morpheme.set_stats(Pair::new(b'b' as u32, b'a' as u32), 7, 0);
    vec![
        ("bpe", Box::new(BpeScorer::new()), 1),
        ("bpe_min2", Box::new(BpeScorer::new()), 2),
        ("length_penalized", Box::new(LengthPenalizedScorer::new(0.5, 1)), 1),
        ("length_rewarded", Box::new(LengthPenalizedScorer::new(-1.0, 1)), 1),
        ("wordpiece", Box::new(WordPieceScorer::new(1.0, 1)), 1),
        ("wordpiece_unsmoothed", Box::new(WordPieceScorer::new(0.0, 1)), 1),
        ("wp_bpe_base", Box::new(WpBpeBaseScorer::new()), 1),
        ("wp_bpe_ft", Box::new(WpBpeFtScorer::new(2)), 2),
        ("wp_bpe_pig", Box::new(WpBpePigScorer::new(1)), 1),
        ("wp_bpe_scp", Box::new(WpBpeScpScorer::new(2, 0.75)), 2),
        ("wp_bpe_npmi", Box::new(WpBpeNpmiScorer::new(2, 0.5)), 2),
        ("wp_bpe_npmi_min1", Box::new(WpBpeNpmiScorer::new(1, 1.0)), 1),
        ("wp_bpe_ppig", Box::new(WpBpePpigScorer::new(2, 0.35)), 2),
        ("mdl", Box::new(MdlScorer::new(24.0, 1)), 1),
        ("mdl_free", Box::new(MdlScorer::new(0.0, 2)), 2),
        ("morpheme_pig", Box::new(morpheme), 1),
        ("noisy_monotone", Box::new(NoisyMonotoneScorer), 1),
        ("arbitrary_deps", Box::new(OddScorer), 1),
    ]
}

#[test]
fn engine_matches_bruteforce_argmax_for_every_scorer() {
    const TRIALS: u64 = 300;
    for (name, scorer, min_freq) in scorers() {
        for seed in 0..TRIALS {
            let mut rng = fastrand::Rng::with_seed(seed);
            let alphabet: &[u8] = [&b"ab"[..], b"abc", b"abcd"][(seed % 3) as usize];
            let corpus = random_corpus(&mut rng, alphabet, 30, 10);
            let target = 256 + 30;

            let model = BottomUpTrainer::new(scorer.as_ref(), target)
                .with_min_frequency(min_freq)
                .train(&corpus)
                .unwrap();
            let (ref_merges, ref_words) = reference_train(scorer.as_ref(), &corpus, target, min_freq);

            assert_eq!(model.merges, ref_merges, "{name}: merge sequence differs from brute force (seed {seed})");
            assert!(model.pretokenize, "{name}: single-segment training words must give a segmented model");

            // Encoding a training word reproduces its training segmentation.
            let words = corpus.iter().filter(|(w, f)| !w.is_empty() && *f > 0);
            for ((word, _), ref_seg) in words.zip(&ref_words) {
                assert_eq!(&model.encode(word), ref_seg, "{name}: encode != training segmentation (seed {seed})");
            }
        }
    }
}

#[test]
fn every_builtin_scorer_declares_narrow_deps() {
    for (name, scorer, _) in scorers() {
        let expected_arbitrary = name == "arbitrary_deps";
        assert_eq!(scorer.deps() == ScoreDeps::Arbitrary, expected_arbitrary, "{name}");
    }
}

#[test]
fn encode_matches_rank_replay_on_unseen_text() {
    for seed in 0..200u64 {
        let mut rng = fastrand::Rng::with_seed(seed);
        let alphabet: &[u8] = if seed % 2 == 0 { b"ab" } else { b"abc" };
        let corpus = random_corpus(&mut rng, alphabet, 25, 12);
        let scorer = WpBpePigScorer::new(1);
        let model = BottomUpTrainer::new(&scorer, 256 + 40).train(&corpus).unwrap();
        let whole = BottomUpModel::new(model.vocab.clone(), model.merges.clone());
        assert!(!whole.pretokenize);

        for _ in 0..10 {
            let len = rng.usize(0..60);
            let text: Vec<u8> = (0..len)
                .map(|_| if rng.u8(0..6) == 0 { b' ' } else { alphabet[rng.usize(0..alphabet.len())] })
                .collect();
            let expected = reference_encode(&model.merges, &text);
            assert_eq!(model.encode(&text), expected, "segmented encode (seed {seed})");
            assert_eq!(whole.encode(&text), expected, "whole-text encode (seed {seed})");
            assert_eq!(model.decode(&expected).unwrap(), text);
        }
    }
}

#[test]
fn encode_matches_rank_replay_for_arbitrary_merge_lists() {
    // Random merge lists contain duplicate tokens ("abc" from (ab,c) and (a,bc)) and pairs merged
    // at several ranks; the heap encoder must still equal replaying every rank in order.
    let mut repeated_pairs = 0;
    for seed in 0..300u64 {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut vocab = tokenisers_core::vocab::Vocabulary::new_byte_level();
        let mut ids: Vec<u32> = b"abc".iter().map(|&b| b as u32).collect();
        let mut merges = Vec::new();
        for _ in 0..40 {
            let (u, v) = (ids[rng.usize(0..ids.len())], ids[rng.usize(0..ids.len())]);
            let mut bytes = vocab.get_bytes(u).unwrap().to_vec();
            bytes.extend_from_slice(vocab.get_bytes(v).unwrap());
            if bytes.len() > 6 {
                continue;
            }
            let w = vocab.add_token(bytes);
            if !ids.contains(&w) {
                ids.push(w);
            }
            merges.push((u, v, w));
        }
        let distinct: std::collections::HashSet<_> = merges.iter().map(|&(u, v, _)| (u, v)).collect();
        repeated_pairs += merges.len() - distinct.len();
        let model = BottomUpModel::new(vocab, merges);
        for _ in 0..20 {
            let text: Vec<u8> = (0..rng.usize(0..40)).map(|_| b"abc"[rng.usize(0..3)]).collect();
            assert_eq!(model.encode(&text), reference_encode(&model.merges, &text), "seed {seed}");
        }
    }
    assert!(repeated_pairs > 0, "generator must exercise repeated pairs");
}

#[test]
fn overlapping_self_pairs_are_counted_and_merged_leftmost() {
    // "aaa" holds two overlapping (a, a) positions but only one merge applies.
    let corpus = vec![(b"aaa".to_vec(), 3u64), (b"aaaa".to_vec(), 1), (b"abab".to_vec(), 2)];
    for (name, scorer, min_freq) in scorers() {
        let model = BottomUpTrainer::new(scorer.as_ref(), 262)
            .with_min_frequency(min_freq)
            .train(&corpus)
            .unwrap();
        let (ref_merges, _) = reference_train(scorer.as_ref(), &corpus, 262, min_freq);
        assert_eq!(model.merges, ref_merges, "{name}");
    }
    let model = BottomUpTrainer::new(&BpeScorer, 257).train(&corpus).unwrap();
    let aa = model.merges[0].2;
    assert_eq!(model.merges[0].0, b'a' as u32);
    assert_eq!(model.encode(b"aaa"), vec![aa, b'a' as u32]);
}

#[test]
fn wordpiece_regression_from_review() {
    // Review repro: after merging (a, b), f(a) drops to 2, so (a, c) = 2/9 must beat (d, e).
    let corpus = vec![
        (b"ab".to_vec(), 10u64),
        (b"ac".to_vec(), 2),
        (b"de".to_vec(), 3),
        (b"d".to_vec(), 3),
        (b"e".to_vec(), 3),
    ];
    let scorer = WordPieceScorer::new(1.0, 1);
    let model = BottomUpTrainer::new(&scorer, 258).train(&corpus).unwrap();
    let pairs: Vec<(u8, u8)> = model.merges.iter().map(|&(u, v, _)| (u as u8, v as u8)).collect();
    assert_eq!(pairs, vec![(b'a', b'b'), (b'a', b'c')]);
}

#[test]
fn training_reaches_target_size_when_pairs_remain() {
    let text = common::pseudo_text(7, 20_000);
    let corpus: Vec<(Vec<u8>, u64)> = {
        let mut counts = std::collections::BTreeMap::new();
        for seg in tokenisers_core::SuperBpeTrainer::segment_text(&text) {
            *counts.entry(seg).or_insert(0u64) += 1;
        }
        counts.into_iter().collect()
    };
    for (name, scorer, min_freq) in scorers() {
        let model = BottomUpTrainer::new(scorer.as_ref(), 400)
            .with_min_frequency(min_freq)
            .train(&corpus)
            .unwrap();
        assert_eq!(model.vocab.len(), 400, "{name} stopped early");
    }
}
