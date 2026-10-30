mod common;

use std::collections::BinaryHeap;
use tokenisers_core::engine::bottom_up::BottomUpTrainer;
use tokenisers_core::index::Pair;
use tokenisers_core::scoring::{
    BpeScorer, LengthPenalizedScorer, MdlScorer, PairContext, PairScorer, ScoreDeps, ScoredPair, WordPieceScorer,
    WpBpeBaseScorer, WpBpeFtScorer, WpBpeNpmiScorer, WpBpePigScorer, WpBpePpigScorer, WpBpeScpScorer,
};

fn dummy_context(freq_uv: u64, freq_u: u64, freq_v: u64, total_tokens: u64, len_u: usize, len_v: usize) -> PairContext {
    PairContext {
        freq_uv,
        freq_u,
        freq_v,
        total_tokens,
        active_vocab_size: 256,
        len_u,
        len_v,
    }
}

#[test]
fn test_bpe_scorer_monotonicity() {
    let scorer = BpeScorer::new();
    let pair = Pair::new(10, 20);

    let ctx1 = dummy_context(100, 200, 200, 1000, 1, 1);
    let ctx2 = dummy_context(99, 200, 200, 1000, 1, 1);

    assert!(scorer.score(pair, &ctx1) > scorer.score(pair, &ctx2));
}

#[test]
fn test_wordpiece_laplace_smoothing_suppresses_singletons() {
    let scorer = WordPieceScorer::new(1.0, 2);
    let pair = Pair::new(1, 2);

    // Rare pair occurring twice with low marginals: freq_uv = 2, freq_u = 2, freq_v = 2
    let ctx_rare = dummy_context(2, 2, 2, 10000, 1, 1);
    let score_rare = scorer.score(pair, &ctx_rare);

    // Frequent collocation: freq_uv = 500, freq_u = 1000, freq_v = 1000
    let ctx_frequent = dummy_context(500, 1000, 1000, 10000, 1, 1);
    let score_frequent = scorer.score(pair, &ctx_frequent);

    assert!(score_frequent > 0.0);
    assert!(score_rare > 0.0);
    // Below threshold (freq < 2) returns NEG_INFINITY
    let ctx_singleton = dummy_context(1, 1, 1, 10000, 1, 1);
    assert_eq!(scorer.score(pair, &ctx_singleton), f64::NEG_INFINITY);
}

#[test]
fn test_mdl_compression_tradeoff() {
    let scorer = MdlScorer::new(24.0, 2);
    let pair = Pair::new(10, 20);

    // Highly frequent pair in 100,000 token corpus gives positive bit savings
    let ctx_frequent = dummy_context(5000, 6000, 6000, 100000, 1, 1);
    let score_frequent = scorer.score(pair, &ctx_frequent);
    assert!(score_frequent > 0.0, "Frequent pair should yield positive Delta-Bits, got {score_frequent}");

    // Rare pair with long token lengths yields negative or low Delta-Bits
    let ctx_rare = dummy_context(2, 2, 2, 100000, 16, 16);
    let score_rare = scorer.score(pair, &ctx_rare);
    assert!(score_rare < 0.0, "Rare long pair should yield negative Delta-Bits, got {score_rare}");
}

#[test]
fn test_length_penalized_beta_effects() {
    let pair = Pair::new(10, 20);
    let ctx_short = dummy_context(100, 100, 100, 1000, 1, 1); // length = 2
    let ctx_long = dummy_context(100, 100, 100, 1000, 4, 4);  // length = 8

    // README: count(uv) / (|u| + |v|)^beta, so beta > 0 penalizes longer merges.
    let scorer_penalty = LengthPenalizedScorer::new(1.0, 1);
    assert_eq!(scorer_penalty.score(pair, &ctx_short), 50.0);
    assert_eq!(scorer_penalty.score(pair, &ctx_long), 12.5);

    // When beta == 0 (standard BPE behavior)
    let scorer_neutral = LengthPenalizedScorer::new(0.0, 1);
    assert_eq!(scorer_neutral.score(pair, &ctx_long), scorer_neutral.score(pair, &ctx_short));

    // When beta < 0 (reward longer tokens)
    let scorer_reward = LengthPenalizedScorer::new(-0.5, 1);
    assert!(scorer_reward.score(pair, &ctx_long) > scorer_reward.score(pair, &ctx_short));
}

#[test]
fn test_scored_pair_deterministic_tie_breaking() {
    let mut heap = BinaryHeap::new();

    // Two pairs with exact same score
    heap.push(ScoredPair {
        score: 10.0,
        pair: Pair::new(50, 60),
        version: 0,
    });
    heap.push(ScoredPair {
        score: 10.0,
        pair: Pair::new(20, 30),
        version: 0,
    });

    // Smaller pair.0 should pop first on tie
    let popped1 = heap.pop().unwrap();
    let popped2 = heap.pop().unwrap();

    assert_eq!(popped1.pair, Pair::new(20, 30));
    assert_eq!(popped2.pair, Pair::new(50, 60));
}

#[test]
fn test_bottom_up_training_all_four_scorers() {
    let corpus = vec![
        (b"the quick brown fox jumps over the lazy dog".to_vec(), 50),
        (b"the lazy dog slept while the quick fox ran".to_vec(), 40),
        (b"reinforcement learning from human feedback".to_vec(), 30),
        (b"transformer attention mechanism tokenization".to_vec(), 20),
    ];

    let bpe_scorer = BpeScorer::new();
    let wp_scorer = WordPieceScorer::new(1.0, 2);
    let mdl_scorer = MdlScorer::new(24.0, 2);
    let lp_scorer = LengthPenalizedScorer::new(0.5, 2);

    let target_vocab = 280;

    let model_bpe = BottomUpTrainer::new(&bpe_scorer, target_vocab).train(&corpus).unwrap();
    let model_wp = BottomUpTrainer::new(&wp_scorer, target_vocab).train(&corpus).unwrap();
    let model_mdl = BottomUpTrainer::new(&mdl_scorer, target_vocab).train(&corpus).unwrap();
    let model_lp = BottomUpTrainer::new(&lp_scorer, target_vocab).train(&corpus).unwrap();

    for model in [&model_bpe, &model_wp, &model_mdl, &model_lp] {
        assert_eq!(model.vocab.len(), target_vocab);
    }

    // Verify all four models decode losslessly
    let test_sentence = "the quick brown fox jumps over the lazy dog";
    for (name, model) in [("BPE", &model_bpe), ("WordPiece", &model_wp), ("MDL", &model_mdl), ("LengthPenalized", &model_lp)] {
        let tokens = model.encode(test_sentence.as_bytes());
        let decoded = model.decode_lossless(&tokens).expect("Lossless decode failed");
        assert_eq!(test_sentence, decoded, "Lossless failure on model {name}");
    }
}

/// Unigram code length N*log2(N) - sum c*log2(c) of a token sequence.
fn unigram_bits(seq: &[u32]) -> f64 {
    let mut counts = std::collections::HashMap::new();
    for &t in seq {
        *counts.entry(t).or_insert(0u64) += 1;
    }
    let xlx = |x: f64| if x > 0.0 { x * x.log2() } else { 0.0 };
    xlx(seq.len() as f64) - counts.values().map(|&c| xlx(c as f64)).sum::<f64>()
}

#[test]
fn test_mdl_matches_recount_including_self_pairs() {
    let scorer = MdlScorer::new(24.0, 1);
    // Finding C5 repro: c_a = 100, c_aa = 40, N = 100 -> corpus gain -55.1 bits.
    let ctx = dummy_context(40, 100, 100, 100, 1, 1);
    let corpus_gain = scorer.delta_bits(Pair::new(1, 1), &ctx) + 24.0 + 16.0;
    assert!((corpus_gain + 55.098).abs() < 1e-2, "{corpus_gain}");
    // A merge that does not shorten the description is ineligible (MDL stopping).
    assert_eq!(scorer.score(Pair::new(1, 1), &ctx), f64::NEG_INFINITY);

    let mut rng = fastrand::Rng::with_seed(3);
    for _ in 0..500 {
        let alphabet = [b'a' as u32, b'b' as u32, b'c' as u32];
        let seq: Vec<u32> = (0..rng.usize(2..60)).map(|_| alphabet[rng.usize(0..3)]).collect();
        let (u, v) = (alphabet[rng.usize(0..3)], alphabet[rng.usize(0..3)]);
        let merged = common::apply_leftmost(&seq, u, v, 300);
        let m = (seq.len() - merged.len()) as u64;
        if m == 0 {
            continue;
        }
        let count = |t: u32| seq.iter().filter(|&&x| x == t).count() as u64;
        let ctx = dummy_context(m, count(u), count(v), seq.len() as u64, 1, 1);
        let recount = unigram_bits(&seq) - unigram_bits(&merged) - (24.0 + 16.0);
        let got = scorer.delta_bits(Pair::new(u, v), &ctx);
        assert!((got - recount).abs() < 1e-9 * (1.0 + recount.abs()), "{u}/{v} on {seq:?}: {got} vs {recount}");
        let score = scorer.score(Pair::new(u, v), &ctx);
        assert_eq!(score, if got > 0.0 { got } else { f64::NEG_INFINITY });
    }
}

#[test]
fn test_wp_bpe_ft_differs_from_wordpiece_on_singleton_pairs() {
    // Unsmoothed PMI loves rare pairs: "xq" occurs once with rare marginals.
    let corpus = vec![
        (b"thethe".to_vec(), 20u64),
        (b"then".to_vec(), 10),
        (b"xq".to_vec(), 1),
        (b"zj".to_vec(), 1),
    ];
    let wordpiece = WordPieceScorer::new(0.0, 1);
    let ft = WpBpeFtScorer::new(2);
    let base = WpBpeBaseScorer::new();
    let train = |s: &dyn PairScorer, mf: u64| BottomUpTrainer::new(s, 262).with_min_frequency(mf).train(&corpus).unwrap().merges;
    let m_wp = train(&wordpiece, 1);
    let m_ft = train(&ft, 2);
    assert_ne!(m_wp, m_ft, "wp_bpe_ft must differ from wordpiece when singleton pairs exist");
    assert!(m_ft.iter().all(|&(u, _, _)| u != b'x' as u32 && u != b'z' as u32));
    // Documented equivalence: unsmoothed wordpiece == wp_bpe_base (same function, no cutoff),
    // while the default Laplace-smoothed wordpiece is a genuinely different objective.
    assert_eq!(m_wp, train(&base, 1));

    // (x,q): f=1, f_x=1, f_q=2 -> 0.5 unsmoothed, 1/6 smoothed; (j,k): f=f_j=f_k=3 -> 1/3, 3/16.
    let corpus = vec![(b"xq".to_vec(), 1u64), (b"q".to_vec(), 1), (b"jk".to_vec(), 3)];
    let first = |s: &dyn PairScorer| BottomUpTrainer::new(s, 257).train(&corpus).unwrap().merges[0];
    assert_eq!(first(&WordPieceScorer::new(0.0, 1)).0, b'x' as u32);
    assert_eq!(first(&WordPieceScorer::new(1.0, 1)).0, b'j' as u32);
}

#[test]
fn test_declared_deps_hold() {
    let pair = Pair::new(1, 2);
    let pair_only: Vec<Box<dyn PairScorer>> = vec![Box::new(BpeScorer), Box::new(LengthPenalizedScorer::new(0.7, 1))];
    let marginals: Vec<Box<dyn PairScorer>> = vec![
        Box::new(WordPieceScorer::new(0.5, 1)),
        Box::new(WpBpeBaseScorer::new()),
        Box::new(WpBpeFtScorer::new(2)),
        Box::new(WpBpeScpScorer::new(2, 0.75)),
    ];
    let monotone: Vec<Box<dyn PairScorer>> = vec![
        Box::new(WpBpePigScorer::new(1)),
        Box::new(WpBpeNpmiScorer::new(2, 0.5)),
        Box::new(WpBpePpigScorer::new(2, 0.35)),
        Box::new(MdlScorer::new(24.0, 1)),
    ];
    let mut rng = fastrand::Rng::with_seed(9);
    for _ in 0..20_000 {
        let f_u = rng.u64(1..5_000);
        let f_v = rng.u64(1..5_000);
        let f_uv = rng.u64(1..=f_u.min(f_v));
        let n = f_u + f_v + rng.u64(0..1_000_000);
        let d = rng.u64(1..=(n - f_u - f_v).max(1)).min(n - 1);
        let (lu, lv) = (rng.usize(1..8), rng.usize(1..8));
        let ctx = dummy_context(f_uv, f_u, f_v, n, lu, lv);
        let ctx_smaller_n = PairContext { total_tokens: n - d, active_vocab_size: 300, ..ctx };
        let ctx_other_marg = PairContext { freq_u: f_u + 1, freq_v: f_v + 3, ..ctx };
        for s in &pair_only {
            assert_eq!(s.deps(), ScoreDeps::PairOnly);
            assert_eq!(s.score(pair, &ctx), s.score(pair, &ctx_other_marg));
            assert_eq!(s.score(pair, &ctx), s.score(pair, &ctx_smaller_n));
        }
        for s in &marginals {
            assert_eq!(s.deps(), ScoreDeps::Marginals);
            assert_eq!(s.score(pair, &ctx), s.score(pair, &ctx_smaller_n));
        }
        for s in &monotone {
            assert_eq!(s.deps(), ScoreDeps::MonotoneTotals);
            let (before, after) = (s.score(pair, &ctx), s.score(pair, &ctx_smaller_n));
            assert!(after <= before, "{}: score rose from {before} to {after} as N shrank ({ctx:?})", s.name());
        }
    }
}

#[test]
fn test_negative_pmi_pairs_stay_eligible() {
    // f(ab) = 1 with f(a) = f(b) = 101, N = 202: PMI < 0. README: npmi keeps its negative value,
    // ppig scores max(0, PMI) = 0; neither may end training early.
    let corpus = vec![(b"ab".to_vec(), 1u64), (b"a".to_vec(), 100), (b"b".to_vec(), 100)];
    let npmi = WpBpeNpmiScorer::new(1, 0.5);
    let ppig = WpBpePpigScorer::new(1, 0.35);
    for scorer in [&npmi as &dyn PairScorer, &ppig] {
        let model = BottomUpTrainer::new(scorer, 257).train(&corpus).unwrap();
        assert_eq!(model.vocab.len(), 257, "{} stopped on a negative-PMI pair", scorer.name());
    }
}
