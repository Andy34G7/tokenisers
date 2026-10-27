use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use tokenisers_core::curricula::super_bpe::{SuperBpeConfig, SuperBpeTrainer};
use tokenisers_core::engine::bottom_up::BottomUpTrainer;
use tokenisers_core::engine::unigram::UnigramTrainer;
use tokenisers_core::export::huggingface::HfTokenizerJson;
use tokenisers_core::index::Pair;
use tokenisers_core::scoring::{
    BpeScorer, LengthPenalizedScorer, MdlScorer, PairContext, PairScorer, WordPieceScorer,
};

// =========================================================================
// 1. Scorer Distinctiveness and Mathematical Correctness
// =========================================================================

#[test]
fn test_scorer_distinctiveness_on_identical_corpus() {
    let corpus: Vec<(Vec<u8>, u64)> = vec![
        (b"common common common common common common common common".to_vec(), 50),
        (b"rare_pair rare_pair rare_pair rare_pair rare_pair".to_vec(), 20),
        (b"supercalifragilisticexpialidocious".to_vec(), 15),
        (b"info_dense_payload info_dense_payload".to_vec(), 30),
        (b"xy xy xy xy xy xy xy xy xy xy".to_vec(), 25),
        (b"ab cd ef gh ij kl mn op qr st uv wx yz".to_vec(), 10),
    ];

    let target_vocab_size = 285; // 256 base + 29 merges

    let bpe = BpeScorer::new();
    let wordpiece = WordPieceScorer::new(1.0, 1);
    let mdl = MdlScorer::new(16.0, 1);
    let lp = LengthPenalizedScorer::new(1.5, 1);

    let model_bpe = BottomUpTrainer::new(&bpe, target_vocab_size).train(&corpus).unwrap();
    let model_wp = BottomUpTrainer::new(&wordpiece, target_vocab_size).train(&corpus).unwrap();
    let model_mdl = BottomUpTrainer::new(&mdl, target_vocab_size).train(&corpus).unwrap();
    let model_lp = BottomUpTrainer::new(&lp, target_vocab_size).train(&corpus).unwrap();

    let merges_bpe: Vec<Pair> = model_bpe.merges.iter().map(|&(u, v, _)| Pair::new(u, v)).collect();
    let merges_wp: Vec<Pair> = model_wp.merges.iter().map(|&(u, v, _)| Pair::new(u, v)).collect();
    let merges_mdl: Vec<Pair> = model_mdl.merges.iter().map(|&(u, v, _)| Pair::new(u, v)).collect();
    let merges_lp: Vec<Pair> = model_lp.merges.iter().map(|&(u, v, _)| Pair::new(u, v)).collect();

    let vocab_bpe: HashSet<Vec<u8>> = model_bpe.vocab.id_to_bytes_map()[256..].iter().cloned().collect();
    let vocab_wp: HashSet<Vec<u8>> = model_wp.vocab.id_to_bytes_map()[256..].iter().cloned().collect();
    let vocab_mdl: HashSet<Vec<u8>> = model_mdl.vocab.id_to_bytes_map()[256..].iter().cloned().collect();
    let vocab_lp: HashSet<Vec<u8>> = model_lp.vocab.id_to_bytes_map()[256..].iter().cloned().collect();

    assert_ne!(merges_bpe, merges_wp, "BPE and WordPiece produced identical merge orders");
    assert_ne!(merges_bpe, merges_mdl, "BPE and MDL produced identical merge orders");
    assert_ne!(merges_bpe, merges_lp, "BPE and LengthPenalized produced identical merge orders");
    assert_ne!(merges_wp, merges_mdl, "WordPiece and MDL produced identical merge orders");
    assert_ne!(merges_wp, merges_lp, "WordPiece and LengthPenalized produced identical merge orders");
    assert_ne!(merges_mdl, merges_lp, "MDL and LengthPenalized produced identical merge orders");

    assert_ne!(vocab_bpe, vocab_wp, "BPE and WordPiece produced identical vocabularies");
    assert_ne!(vocab_bpe, vocab_mdl, "BPE and MDL produced identical vocabularies");
    assert_ne!(vocab_bpe, vocab_lp, "BPE and LengthPenalized produced identical vocabularies");
    assert_ne!(vocab_wp, vocab_mdl, "WordPiece and MDL produced identical vocabularies");
    assert_ne!(vocab_wp, vocab_lp, "WordPiece and LengthPenalized produced identical vocabularies");
    assert_ne!(vocab_mdl, vocab_lp, "MDL and LengthPenalized produced identical vocabularies");
}

#[test]
fn test_mdl_favors_high_information_density() {
    let mdl = MdlScorer::new(24.0, 1);

    let ctx_dense = PairContext {
        freq_uv: 100,
        freq_u: 100,
        freq_v: 100,
        total_tokens: 1000,
        active_vocab_size: 256,
        len_u: 1,
        len_v: 1,
    };

    let ctx_sparse = PairContext {
        freq_uv: 2,
        freq_u: 2,
        freq_v: 2,
        total_tokens: 1000,
        active_vocab_size: 256,
        len_u: 1,
        len_v: 1,
    };

    let ctx_independent = PairContext {
        freq_uv: 50,
        freq_u: 500,
        freq_v: 500,
        total_tokens: 1000,
        active_vocab_size: 256,
        len_u: 1,
        len_v: 1,
    };

    let score_dense = mdl.score(Pair::new(1, 2), &ctx_dense);
    let score_sparse = mdl.score(Pair::new(3, 4), &ctx_sparse);
    let score_independent = mdl.score(Pair::new(5, 6), &ctx_independent);

    assert!(score_dense > 0.0, "High density pair should yield positive compression bits");
    assert!(score_sparse < 0.0, "Sparse pair should yield negative score due to vocab overhead");
    assert!(score_dense > score_independent, "High-density pair must outscore independent pair");
}

/// =========================================================================
/// 2. SuperBPE 2-Stage Transition
/// =========================================================================

#[test]
fn test_superbpe_stage1_strictly_prevents_cross_word_merges() {
    let lines = vec![
        "abcdefgh ijklmnop qrstuvwx yzabcdef",
        "abcdefgh ijklmnop qrstuvwx yzabcdef",
        "abcdefgh ijklmnop qrstuvwx yzabcdef",
    ];

    let v_switch = 265;
    let v_final = 265;
    let config = SuperBpeConfig::new(v_switch, v_final).unwrap();
    let scorer = BpeScorer::new();
    let trainer = SuperBpeTrainer::new(config, &scorer);

    let model = trainer.train(&lines).expect("SuperBPE training failed");

    for (id, bytes) in model.vocab.id_to_bytes_map().iter().enumerate() {
        if id < 256 {
            continue;
        }
        let has_whitespace = bytes.iter().any(|&b| b == b' ' || b == b'\t' || b == b'\n');
        assert!(
            !has_whitespace,
            "Stage 1 learned token containing whitespace: {:?} (string: {:?})",
            bytes,
            String::from_utf8_lossy(bytes)
        );
    }
}

/// CHALLENGE FAILURE 1: SuperBPE Stage 2 cannot learn full cross-word collocations
/// because boundary junctions store initial node indices which get tombstoned during Stage 1.
#[test]
fn test_superbpe_stage2_fails_to_form_collocation_when_tail_merged() {
    let lines = vec![
        "ab cd ab cd ab cd ab cd ab cd",
        "ab cd ab cd ab cd ab cd ab cd",
        "ab cd ab cd ab cd ab cd ab cd",
    ];

    let v_switch = 258;
    let v_final = 265;
    let config = SuperBpeConfig::new(v_switch, v_final).unwrap();
    let scorer = BpeScorer::new();
    let trainer = SuperBpeTrainer::new(config, &scorer);

    let model = trainer.train(&lines).expect("SuperBPE training failed");

    let learned_tokens: Vec<String> = model.vocab.id_to_bytes_map()[256..]
        .iter()
        .map(|b| String::from_utf8_lossy(b).to_string())
        .collect();

    // Check if full collocation "ab cd" or left-linked collocation "ab " was formed
    let learned_full_collocation = learned_tokens.iter().any(|s| s == "ab cd" || s == "ab ");
    assert!(
        learned_full_collocation,
        "SuperBPE Stage 2 failed to form full collocation 'ab cd' or 'ab ' due to tombstoned boundary junction: learned {:?}",
        learned_tokens
    );
}

/// CHALLENGE FAILURE 2: SuperBPE Stage 1 pair exhaustion causes an infinite 100% CPU loop
/// because `vocab.add_token(vec![0x20])` returns existing byte 32 without incrementing vocab.len().
#[test]
fn test_superbpe_stage1_pair_exhaustion_infinite_loop() {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        // "hello" has only 4 possible merges ('he', 'el', 'll', 'lo')
        // Asking for v_switch = 270 requires 14 merges, which cannot be satisfied.
        let lines = vec!["hello"];
        let config = SuperBpeConfig::new(270, 280).unwrap();
        let scorer = BpeScorer::new();
        let trainer = SuperBpeTrainer::new(config, &scorer);
        let _ = trainer.train(&lines);
        let _ = tx.send(());
    });

    let res = rx.recv_timeout(Duration::from_millis(500));
    assert!(
        res.is_ok(),
        "SuperBPE hung in an infinite loop when Stage 1 intra-word pairs were exhausted before v_switch!"
    );
}

// =========================================================================
// 3. HuggingFace Export Artifact Generation for Python Validation
// =========================================================================

#[test]
fn export_models_for_python_verification() {
    let corpus = vec![
        (b"the quick brown fox jumps over the lazy dog".to_vec(), 50),
        (b"the lazy dog slept while the quick fox ran".to_vec(), 40),
        (b"reinforcement learning from human feedback".to_vec(), 30),
        (b"transformer attention mechanism tokenization".to_vec(), 25),
    ];

    let export_dir = Path::new("/tmp/tokenisers_hf_test");
    let _ = fs::create_dir_all(export_dir.join("bpe"));
    let _ = fs::create_dir_all(export_dir.join("unigram"));

    let bpe_scorer = BpeScorer::new();
    let bpe_model = BottomUpTrainer::new(&bpe_scorer, 280).train(&corpus).unwrap();
    let hf_bpe = HfTokenizerJson::from_bpe_model(&bpe_model, None);
    hf_bpe.export_to_file(export_dir.join("bpe")).expect("BPE export failed");

    let unigram_trainer = UnigramTrainer::new(280, 0.2);
    let unigram_model = unigram_trainer.train(&corpus).unwrap();
    let hf_unigram = HfTokenizerJson::from_unigram_model(&unigram_model, None);
    hf_unigram.export_to_file(export_dir.join("unigram")).expect("Unigram export failed");

    assert!(export_dir.join("bpe/tokenizer.json").exists());
    assert!(export_dir.join("unigram/tokenizer.json").exists());
}
