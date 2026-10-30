use std::collections::HashMap;

use tokenisers_core::engine::bottom_up::BottomUpTrainer;
use tokenisers_core::engine::unigram::{CompactTrie, ViterbiDecoder};
use tokenisers_core::inference::bpe_dropout::BpeDropoutEncoder;
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

    let encoder = BpeDropoutEncoder::new(&model.merges);
    let text = "the quick brown fox jumps over the lazy dog";
    let bytes = text.as_bytes();

    // 1. p = 0.0 matches greedy BPE exactly
    let greedy_tokens = model.encode(bytes);
    let dropout_p0 = encoder.encode(bytes, 0.0, None);
    assert_eq!(greedy_tokens, dropout_p0);

    // 2. p = 1.0 produces 100% single-byte base tokens
    let dropout_p1 = encoder.encode(bytes, 1.0, None);
    assert_eq!(dropout_p1.len(), bytes.len());
    for (i, &tok) in dropout_p1.iter().enumerate() {
        assert_eq!(tok, bytes[i] as u32);
    }

    // 3. For 0 < p < 1, outputs vary with seed but all decode losslessly
    let sample1 = encoder.encode(bytes, 0.4, Some(42));
    let sample2 = encoder.encode(bytes, 0.4, Some(12345));

    // Diverse segmentations
    assert_ne!(sample1, sample2);

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

// ---------------------------------------------------------------------------------------------
// Unigram samplers against the exact (enumerated) posterior.
// ---------------------------------------------------------------------------------------------

fn sampler_vocab() -> Vec<(Vec<u8>, u32, f64)> {
    vec![
        (b"a".to_vec(), 1, -2.0),
        (b"b".to_vec(), 2, -2.3),
        (b"c".to_vec(), 3, -1.8),
        (b"ab".to_vec(), 4, -2.9),
        (b"bc".to_vec(), 5, -3.1),
        (b"abc".to_vec(), 6, -4.2),
        (b"ca".to_vec(), 7, -3.5),
    ]
}

/// All segmentations of `text` with their total log-probability.
fn enumerate_segmentations(text: &[u8], vocab: &[(Vec<u8>, u32, f64)]) -> Vec<(Vec<u32>, f64)> {
    if text.is_empty() {
        return vec![(Vec::new(), 0.0)];
    }
    let mut out = Vec::new();
    for (bytes, id, lp) in vocab {
        if text.starts_with(bytes) {
            for (mut rest, rest_lp) in enumerate_segmentations(&text[bytes.len()..], vocab) {
                rest.insert(0, *id);
                out.push((rest, lp + rest_lp));
            }
        }
    }
    out
}

/// Normalised weights P(s)^(1/T) over the given segmentations.
fn tempered(segs: &[(Vec<u32>, f64)], temperature: f64) -> HashMap<Vec<u32>, f64> {
    let max = segs.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = segs.iter().map(|s| ((s.1 - max) / temperature).exp()).sum();
    segs.iter()
        .map(|(seg, lp)| (seg.clone(), ((lp - max) / temperature).exp() / z))
        .collect()
}

fn assert_frequencies_match(name: &str, samples: &[Vec<u32>], expected: &HashMap<Vec<u32>, f64>) {
    let n = samples.len() as f64;
    let mut counts: HashMap<Vec<u32>, f64> = HashMap::new();
    for s in samples {
        *counts.entry(s.clone()).or_insert(0.0) += 1.0;
    }
    for seg in counts.keys() {
        assert!(expected.contains_key(seg), "[{name}] sampled impossible segmentation {seg:?}");
    }
    for (seg, &p) in expected {
        let got = counts.get(seg).copied().unwrap_or(0.0) / n;
        let tol = 5.0 * (p * (1.0 - p) / n).sqrt() + 1e-3;
        assert!((got - p).abs() <= tol, "[{name}] {seg:?}: sampled {got:.4}, exact {p:.4}");
    }
}

#[test]
fn ffbs_samples_the_exact_tempered_posterior() {
    let vocab = sampler_vocab();
    let trie = CompactTrie::build(&vocab);
    let sampler = UnigramSampler::new(&trie);
    let text = b"abcabc";
    let segs = enumerate_segmentations(text, &vocab);
    assert!(segs.len() > 10);
    for temperature in [1.0, 2.5, 0.5] {
        let samples: Vec<Vec<u32>> =
            (0..40_000).map(|seed| sampler.sample_ffbs(text, temperature, Some(seed))).collect();
        assert_frequencies_match(&format!("ffbs T={temperature}"), &samples, &tempered(&segs, temperature));
    }
}

#[test]
fn kbest_samples_the_top_k_renormalised() {
    let vocab = sampler_vocab();
    let trie = CompactTrie::build(&vocab);
    let sampler = UnigramSampler::new(&trie);
    let text = b"abcabc";
    let mut segs = enumerate_segmentations(text, &vocab);
    segs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (k, temperature) in [(3, 1.0), (5, 2.0)] {
        let top_k = &segs[..k];
        let samples: Vec<Vec<u32>> =
            (0..40_000).map(|seed| sampler.sample_kbest(text, k, temperature, Some(seed))).collect();
        assert_frequencies_match(&format!("k={k} T={temperature}"), &samples, &tempered(top_k, temperature));
    }
    // k = 1 is the Viterbi path.
    assert_eq!(sampler.sample_kbest(text, 1, 1.0, None), segs[0].0);
}

#[test]
fn sampler_lattice_never_crosses_whitespace_segments() {
    // "a b" with a token spanning the space: encode, FFBS and k-best must all ignore it.
    let vocab = vec![
        (b"a".to_vec(), 1, -5.0),
        (b" ".to_vec(), 2, -5.0),
        (b"b".to_vec(), 3, -5.0),
        (b"a b".to_vec(), 4, -0.1),
    ];
    let trie = CompactTrie::build(&vocab);
    let sampler = UnigramSampler::new(&trie);
    for seed in 0..50 {
        assert_eq!(sampler.sample_ffbs(b"a b", 1.0, Some(seed)), vec![1, 2, 3]);
        assert_eq!(sampler.sample_kbest(b"a b", 4, 1.0, Some(seed)), vec![1, 2, 3]);
    }
}

// ---------------------------------------------------------------------------------------------
// Byte-entropy patching: BLT entropy rule with backoff.
// ---------------------------------------------------------------------------------------------

fn boundaries(patches: &[Vec<u8>]) -> Vec<usize> {
    patches
        .iter()
        .scan(0, |pos, p| {
            *pos += p.len();
            Some(*pos)
        })
        .collect()
}

fn trained(order: usize, tau: f64, min_len: usize, max_len: usize, corpus: &[u8]) -> ByteEntropyPatcher {
    let mut p = ByteEntropyPatcher::new(ByteEntropyConfig {
        context_order: order,
        tau_threshold: tau,
        min_len,
        max_len,
    });
    p.train_transitions(corpus);
    p
}

#[test]
fn byte_entropy_first_patch_is_not_forced_at_min_len() {
    // Contexts shorter than `order` at the stream start back off to shorter contexts instead of
    // reading as "unseen = 8 bits" and forcing a cut at min_len.
    let patcher = trained(3, 1.0, 2, 32, &b"abc".repeat(200));
    let stream = b"abcabcabc";
    assert_eq!(patcher.segment_stream(stream), vec![stream.to_vec()]);
}

#[test]
fn byte_entropy_cut_depends_only_on_history_not_on_the_next_byte() {
    // Entropy H(next | history) is a property of the history: changing the byte that follows
    // must not move any boundary before it (a surprise rule -log2 P(actual next) would).
    let patcher = trained(3, 1.0, 2, 32, &b"abc".repeat(200));
    let predictable = patcher.segment_stream(b"abcabcabca");
    let surprising = patcher.segment_stream(b"abcabcabcZ");
    assert_eq!(boundaries(&predictable), boundaries(&surprising));
}

#[test]
fn byte_entropy_cuts_after_high_entropy_contexts() {
    // After 'x' any of 26 letters follows (H ~ 4.7 bits); everything else is deterministic.
    let corpus: Vec<u8> = (0..200)
        .flat_map(|_| (b'a'..=b'z').flat_map(|c| [b'x', c, b'.']))
        .collect();
    let patcher = trained(1, 3.0, 1, 16, &corpus);
    assert!(patcher.next_byte_entropy(b"x") > 4.0);
    assert!(patcher.next_byte_entropy(b"m") < 1.0);
    let patches = patcher.segment_stream(b"xm.xq.");
    assert_eq!(patches, vec![b"x".to_vec(), b"m.x".to_vec(), b"q.".to_vec()]);
}

#[test]
fn byte_entropy_untrained_patcher_is_reported_and_chunks_at_min_len() {
    let patcher = ByteEntropyPatcher::new(ByteEntropyConfig {
        context_order: 3,
        tau_threshold: 4.2,
        min_len: 3,
        max_len: 16,
    });
    assert!(!patcher.is_trained());
    assert_eq!(patcher.next_byte_entropy(b"abc"), 8.0);
    assert_eq!(boundaries(&patcher.segment_stream(b"abcdefgh")), vec![3, 6, 8]);
    assert!(trained(3, 4.2, 3, 16, b"ab").is_trained());
}
