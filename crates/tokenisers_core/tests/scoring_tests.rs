use std::collections::BinaryHeap;
use tokenisers_core::engine::bottom_up::BottomUpTrainer;
use tokenisers_core::index::Pair;
use tokenisers_core::scoring::{
    BpeScorer, LengthPenalizedScorer, MdlScorer, PairContext, PairScorer, ScoredPair, WordPieceScorer,
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

    // When beta > 0 (reward longer tokens)
    let scorer_reward = LengthPenalizedScorer::new(1.0, 1);
    assert!(scorer_reward.score(pair, &ctx_long) > scorer_reward.score(pair, &ctx_short));

    // When beta == 0 (standard BPE behavior)
    let scorer_neutral = LengthPenalizedScorer::new(0.0, 1);
    assert_eq!(scorer_neutral.score(pair, &ctx_long), scorer_neutral.score(pair, &ctx_short));

    // When beta < 0 (penalize longer tokens)
    let scorer_penalty = LengthPenalizedScorer::new(-0.5, 1);
    assert!(scorer_penalty.score(pair, &ctx_short) > scorer_penalty.score(pair, &ctx_long));
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

    assert!(model_bpe.vocab.len() >= 256);
    assert!(model_wp.vocab.len() >= 256);
    assert!(model_mdl.vocab.len() >= 256);
    assert!(model_lp.vocab.len() >= 256);

    // Verify all four models decode losslessly
    let test_sentence = "the quick brown fox jumps over the lazy dog";
    for (name, model) in [("BPE", &model_bpe), ("WordPiece", &model_wp), ("MDL", &model_mdl), ("LengthPenalized", &model_lp)] {
        let tokens = model.encode(test_sentence.as_bytes());
        let decoded = model.decode_lossless(&tokens).expect("Lossless decode failed");
        assert_eq!(test_sentence, decoded, "Lossless failure on model {name}");
    }
}
