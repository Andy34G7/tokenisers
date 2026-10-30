use std::process::Command;

use tokenisers_core::engine::unigram::{
    logsumexp2, CompactTrie, UnigramTrainer, ViterbiDecoder,
};
use tokenisers_core::export::HfTokenizerJson;

#[test]
fn test_logsumexp2_numerical_stability() {
    // ln(exp(-1000) + exp(-1000)) = -1000 + ln(2)
    let val = logsumexp2(-1000.0, -1000.0);
    let expected = -1000.0 + 2.0f64.ln();
    assert!((val - expected).abs() < 1e-10, "logsumexp2 failed stability check: {val} vs {expected}");

    // Underflow guard: a much larger than b
    assert_eq!(logsumexp2(0.0, -100.0), 0.0);
    assert_eq!(logsumexp2(f64::NEG_INFINITY, -5.0), -5.0);
    assert_eq!(logsumexp2(-5.0, f64::NEG_INFINITY), -5.0);
}

#[test]
fn test_compact_trie_prefix_matching() {
    let vocab = vec![
        (b"h".to_vec(), 1, -1.0),
        (b"he".to_vec(), 2, -2.0),
        (b"hel".to_vec(), 3, -3.0),
        (b"hello".to_vec(), 4, -4.0),
        (b"world".to_vec(), 5, -5.0),
        // Multilingual UTF-8 "你", "你好"
        ("你".as_bytes().to_vec(), 6, -1.5),
        ("你好".as_bytes().to_vec(), 7, -2.5),
    ];

    let trie = CompactTrie::build(&vocab);

    // Search on "hello world"
    let matches = trie.common_prefix_search(b"hello world");
    let match_lens: Vec<usize> = matches.iter().map(|m| m.0).collect();
    let match_ids: Vec<u32> = matches.iter().map(|m| m.1).collect();

    assert_eq!(match_lens, vec![1, 2, 3, 5]);
    assert_eq!(match_ids, vec![1, 2, 3, 4]);

    // Search on Chinese "你好世界"
    let matches_cn = trie.common_prefix_search("你好世界".as_bytes());
    assert_eq!(matches_cn.len(), 2);
    assert_eq!(matches_cn[0].1, 6); // "你"
    assert_eq!(matches_cn[1].1, 7); // "你好"
}

#[test]
fn test_viterbi_candidate_skipping() {
    let vocab = vec![
        (b"a".to_vec(), 1, -2.0),
        (b"b".to_vec(), 2, -2.0),
        (b"ab".to_vec(), 3, -0.5), // High probability compound
    ];
    let trie = CompactTrie::build(&vocab);

    // Default Viterbi should pick token 3 ("ab")
    let (best_score, best_path) = ViterbiDecoder::decode(b"ab", &trie, None).expect("Viterbi decode failed");
    assert_eq!(best_path, vec![3]);
    assert!((best_score - (-0.5)).abs() < 1e-6);

    // With token 3 skipped, Viterbi must fall back to ["a", "b"]
    let (alt_score, alt_path) = ViterbiDecoder::decode(b"ab", &trie, Some(3)).expect("Alternative path failed");
    assert_eq!(alt_path, vec![1, 2]);
    assert!((alt_score - (-4.0)).abs() < 1e-6);
}

#[test]
fn test_unigram_training_and_base_byte_protection() {
    let corpus = vec![
        (b"natural language processing with neural networks".to_vec(), 40),
        (b"subword tokenization and language model training".to_vec(), 35),
        (b"deep learning representations and embeddings".to_vec(), 30),
    ];

    let target_size = 280;
    let trainer = UnigramTrainer::new(target_size, 0.20);
    let model = trainer.train(&corpus).expect("Unigram training failed");

    // The seed set (~1000 substrings) is far larger than the target, so pruning must land on it.
    assert_eq!(model.vocab.len(), target_size);

    // Strict base 256 byte protection: all single bytes 0..=255 MUST be present
    for b in 0..=255u8 {
        assert!(
            model.vocab.get_id(&[b]).is_some(),
            "Base byte 0x{b:02X} was pruned from Unigram vocabulary!"
        );
    }

    // Verify losslessness on multiple strings
    let test_strings = [
        "natural language processing",
        "deep neural networks",
        "unseen random word: zyxwvutsrqponmlkjihgfedcba",
        "emoji support: 🚀🔥💯",
    ];

    for text in test_strings {
        let tokens = model.encode(text.as_bytes());
        let decoded = model.decode_lossless(&tokens).expect("Unigram decode failed");
        assert_eq!(text, decoded);
    }
}

/// Many equal counts, so every tie-break (seed order, id assignment, pruning) is exercised.
fn tie_heavy_corpus() -> Vec<(Vec<u8>, u64)> {
    let words = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "iota", "kappa", "lambda",
        "mu", "omicron", "sigma", "upsilon", "omega", "日本語", "テキスト", "한국어", "🚀🔥",
        " ", "  ", "\n", "\t",
    ];
    words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_bytes().to_vec(), 3 + (i % 3) as u64))
        .collect()
}

/// Everything training decides: ids, bytes, exact scores (via the export) and encodings.
fn unigram_fingerprint(reverse_corpus: bool) -> String {
    let mut corpus = tie_heavy_corpus();
    if reverse_corpus {
        corpus.reverse();
    }
    let model = UnigramTrainer::new(300, 0.2).train(&corpus).unwrap();
    let json = HfTokenizerJson::from_unigram_model(&model, None).to_json_string().unwrap();
    let ids: Vec<Vec<u32>> = ["alphabet soup", "テキスト 日本語", "omega\tsigma  mu"]
        .iter()
        .map(|s| model.encode(s.as_bytes()))
        .collect();
    format!("{json}\n{ids:?}")
}

/// Helper run in a child process by `unigram_training_is_deterministic_across_processes`.
#[test]
fn unigram_determinism_child() {
    if let Ok(order) = std::env::var("UNIGRAM_DETERMINISM_CHILD") {
        println!("FINGERPRINT<<{}>>FINGERPRINT", unigram_fingerprint(order == "reverse"));
    }
}

/// Hash seeds (ahash/std RandomState) differ per process, so in-process repetition cannot catch
/// hash-order dependence. Train in fresh processes, with different corpus order and thread
/// counts, and require byte-identical vocab, scores and ids.
#[test]
fn unigram_training_is_deterministic_across_processes() {
    let exe = std::env::current_exe().unwrap();
    let child = |order: &str, threads: &str| {
        let out = Command::new(&exe)
            .args(["unigram_determinism_child", "--exact", "--nocapture", "--test-threads=1"])
            .env("UNIGRAM_DETERMINISM_CHILD", order)
            .env("RAYON_NUM_THREADS", threads)
            .output()
            .unwrap();
        assert!(out.status.success(), "child failed: {}", String::from_utf8_lossy(&out.stderr));
        let stdout = String::from_utf8(out.stdout).unwrap();
        let start = stdout.find("FINGERPRINT<<").expect("child printed no fingerprint") + 13;
        let end = stdout.find(">>FINGERPRINT").unwrap();
        stdout[start..end].to_string()
    };
    let here = unigram_fingerprint(false);
    assert_eq!(child("forward", "1"), here, "differs across processes");
    assert_eq!(child("forward", "4"), here, "differs with thread count");
    assert_eq!(child("reverse", "3"), here, "depends on corpus order");
}
