use tokenisers_core::engine::bottom_up::BottomUpTrainer;
use tokenisers_core::engine::unigram::{CompactTrie, ViterbiDecoder};
use tokenisers_core::inference::byte_entropy::{ByteEntropyConfig, ByteEntropyPatcher};
use tokenisers_core::inference::unigram_sample::UnigramSampler;
use tokenisers_core::scoring::BpeScorer;

#[test]
fn test_bpe_dropout_determinism_vs_diversity() {
    let corpus = vec![
        (b"the quick brown fox jumps over the lazy dog".to_vec(), 50),
        (b"the lazy dog slept while the quick fox ran".to_vec(), 40),
    ];
    let scorer = BpeScorer::new();
    let model = BottomUpTrainer::new(&scorer, 275).train(&corpus).unwrap();

    let text = "the quick brown fox jumps over the lazy dog";
    let bytes = text.as_bytes();

    // 1. p = 0.0 matches greedy BPE exactly
    let greedy_tokens = model.encode(bytes);
    let dropout_p0 = model.encode_with_dropout(bytes, 0.0, None);
    assert_eq!(greedy_tokens, dropout_p0);

    // 2. p = 1.0 produces 100% single-byte base tokens
    let dropout_p1 = model.encode_with_dropout(bytes, 1.0, None);
    assert_eq!(dropout_p1.len(), bytes.len());
    for (i, &tok) in dropout_p1.iter().enumerate() {
        assert_eq!(tok, bytes[i] as u32);
    }

    // 3. For 0 < p < 1, outputs vary with seed but all decode losslessly
    let sample1 = model.encode_with_dropout(bytes, 0.4, Some(42));
    let sample2 = model.encode_with_dropout(bytes, 0.4, Some(12345));

    // Diverse segmentations (two fixed seeds may legitimately coincide, so look at many)
    let distinct: std::collections::HashSet<Vec<u32>> =
        (0..20).map(|seed| model.encode_with_dropout(bytes, 0.4, Some(seed))).collect();
    assert!(distinct.len() > 1);

    // Both decode losslessly
    let decoded1 = model.decode_lossless(&sample1).unwrap();
    let decoded2 = model.decode_lossless(&sample2).unwrap();
    assert_eq!(text, decoded1);
    assert_eq!(text, decoded2);
}

#[test]
fn test_unigram_sampling_ffbs_and_kbest() {
    let vocab = vec![
        (b"t".to_vec(), 1, -1.0),
        (b"h".to_vec(), 2, -1.0),
        (b"e".to_vec(), 3, -1.0),
        (b"th".to_vec(), 4, -0.5),
        (b"he".to_vec(), 5, -0.6),
        (b"the".to_vec(), 6, -0.2),
    ];

    let trie = CompactTrie::build(&vocab);
    let sampler = UnigramSampler::new(&trie);
    let text = b"the";

    // Viterbi 1-best should pick token 6 ("the")
    let (_, viterbi_path) = ViterbiDecoder::decode(text, &trie, None).unwrap();
    assert_eq!(viterbi_path, vec![6]);

    // k = 1 should exactly match Viterbi 1-best
    let kbest_1 = sampler.sample_kbest(text, 1, 1.0, None);
    assert_eq!(kbest_1, viterbi_path);

    // Low temperature FFBS converges to 1-best
    let low_temp = sampler.sample_ffbs(text, 0.001, Some(42));
    assert_eq!(low_temp, viterbi_path);

    // High temperature FFBS samples valid alternative decompositions like ["th", "e"] or ["t", "he"]
    let mut sampled_lengths = std::collections::HashSet::new();
    for seed in 0..50 {
        let sample = sampler.sample_ffbs(text, 5.0, Some(seed));
        sampled_lengths.insert(sample.len());
    }
    // Should have sampled both single-token "the" (len 1) and subwords (len 2 or 3)
    assert!(sampled_lengths.len() > 1);
}

#[test]
fn test_byte_entropy_guardrails_and_reconstruction() {
    let config = ByteEntropyConfig {
        context_order: 2,
        tau_threshold: 3.5,
        min_len: 2,
        max_len: 8,
    };

    let mut patcher = ByteEntropyPatcher::new(config);
    let train_corpus = b"The quick brown fox jumps over the lazy dog. Repetitive text repetitive text.";
    patcher.train_transitions(train_corpus);

    let test_stream = b"The quick brown fox jumps over the lazy dog with unseen phrases 1234567890!";
    let patches = patcher.segment_stream(test_stream);

    assert!(!patches.is_empty());

    // Verify guardrails on all patches except potentially the last one
    for (i, patch) in patches.iter().enumerate() {
        if i + 1 < patches.len() {
            assert!(
                patch.len() >= 2,
                "Patch {i} length {} violates min_len 2",
                patch.len()
            );
        }
        assert!(
            patch.len() <= 8,
            "Patch {i} length {} violates max_len 8",
            patch.len()
        );
    }

    // Verify exact byte losslessness: concatenation of all patches equals original stream
    let reconstructed: Vec<u8> = patches.into_iter().flatten().collect();
    assert_eq!(test_stream.to_vec(), reconstructed);
}
