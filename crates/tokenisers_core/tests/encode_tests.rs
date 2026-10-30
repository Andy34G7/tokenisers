//! Encoder behaviour: complexity, segment cache equivalence, BPE-dropout, special tokens.
mod common;

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use common::{pseudo_text, reference_encode};
use tokenisers_core::curricula::super_bpe::{SuperBpeConfig, SuperBpeTrainer};
use tokenisers_core::engine::bottom_up::{BottomUpModel, BottomUpTrainer};
use tokenisers_core::scoring::BpeScorer;
use tokenisers_core::vocab::TokenId;

fn segment_counts(text: &str) -> Vec<(Vec<u8>, u64)> {
    let mut counts = BTreeMap::new();
    for seg in SuperBpeTrainer::segment_text(text) {
        *counts.entry(seg).or_insert(0u64) += 1;
    }
    counts.into_iter().collect()
}

fn trained_model(vocab_size: usize) -> BottomUpModel {
    let text = pseudo_text(1, 300_000);
    BottomUpTrainer::new(&BpeScorer, vocab_size).train(&segment_counts(&text)).unwrap()
}

/// SuperBPE (stage 2 reached): tokens span whitespace, so the model encodes whole text.
fn super_bpe_model() -> BottomUpModel {
    let text = pseudo_text(1, 300_000);
    let lines: Vec<&str> = text.split('\n').collect();
    let config = SuperBpeConfig::new(1200, 1600).unwrap();
    let model = SuperBpeTrainer::new(config, &BpeScorer).train(&lines).unwrap();
    assert!(!model.pretokenize);
    model
}

fn best_of(runs: usize, f: impl Fn()) -> Duration {
    (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .min()
        .unwrap()
}

#[test]
fn segmented_and_whole_text_encoding_agree() {
    let model = trained_model(1500);
    assert!(model.pretokenize);
    let whole = BottomUpModel::new(model.vocab.clone(), model.merges.clone());
    let text = pseudo_text(99, 50_000);
    let ids = model.encode(text.as_bytes());
    assert_eq!(ids, whole.encode(text.as_bytes()));
    assert_eq!(ids, reference_encode(&model.merges, text.as_bytes()));
    assert!(ids.len() * 2 < text.len(), "learned merges must actually compress");
    assert_eq!(model.decode_lossless(&ids).unwrap(), text);

    // Invalid UTF-8 is segmented byte-wise (invalid bytes count as non-whitespace).
    let mut raw = text.as_bytes()[..2000].to_vec();
    raw.extend_from_slice(&[0xff, 0xc3, b'x', 0x80]);
    assert_eq!(model.encode(&raw), reference_encode(&model.merges, &raw));
}

#[test]
fn super_bpe_encoding_matches_rank_replay() {
    let model = super_bpe_model();
    let cross = model.merges.iter().filter(|&&(_, _, w)| {
        let b = model.vocab.get_bytes(w).unwrap();
        b.iter().any(u8::is_ascii_whitespace) && !b.iter().all(u8::is_ascii_whitespace)
    });
    assert!(cross.count() > 100, "stage 2 must learn cross-whitespace tokens");
    let text = pseudo_text(99, 50_000);
    let ids = model.encode(text.as_bytes());
    assert_eq!(ids, reference_encode(&model.merges, text.as_bytes()));
    assert_eq!(model.decode_lossless(&ids).unwrap(), text);
}

#[test]
fn encode_time_scales_linearly() {
    // Review finding: the whole-text (SuperBPE) encoder took 12.5x-16.7x time for 8x input
    // (33x for 16x here); linear encoding takes ~14-16x.
    let model = trained_model(1500);
    let whole = BottomUpModel::new(model.vocab.clone(), model.merges.clone());
    let superbpe = super_bpe_model();
    let small = pseudo_text(5, 128 * 1024);
    let large = pseudo_text(5, 16 * 128 * 1024);
    for (name, m) in [("segmented", &model), ("whole-text", &whole), ("superbpe", &superbpe)] {
        let t_small = best_of(5, || {
            std::hint::black_box(m.encode(small.as_bytes()));
        });
        let t_large = best_of(5, || {
            std::hint::black_box(m.encode(large.as_bytes()));
        });
        let ratio = t_large.as_secs_f64() / t_small.as_secs_f64();
        assert!(ratio <= 22.0, "{name}: 16x input took {ratio:.1}x time ({t_small:?} -> {t_large:?})");
    }
}

/// Run with `cargo test --release --test encode_tests -- --ignored --nocapture`.
#[test]
#[ignore]
fn report_encode_throughput() {
    let model = trained_model(8000);
    let whole = BottomUpModel::new(model.vocab.clone(), model.merges.clone());
    let text = pseudo_text(11, 1 << 20);
    let superbpe = super_bpe_model();
    for (name, m) in [("segmented+cache", &model), ("whole-text", &whole), ("superbpe", &superbpe)] {
        let t = best_of(5, || {
            std::hint::black_box(m.encode(text.as_bytes()));
        });
        println!("{name}: {:.1} MB/s on 1 MiB", 1.0 / t.as_secs_f64());
    }
}

#[test]
fn dropout_p0_equals_encode_and_p1_gives_bytes() {
    let model = trained_model(800);
    let text = pseudo_text(3, 5000);
    let bytes = text.as_bytes();
    let greedy = model.encode(bytes);
    assert_eq!(model.encode_with_dropout(bytes, 0.0, Some(1)), greedy);
    assert_eq!(model.encode_with_dropout(bytes, -1.0, None), greedy);
    assert_eq!(model.encode_with_dropout(bytes, f32::NAN, None), greedy);
    let raw: Vec<TokenId> = bytes.iter().map(|&b| b as TokenId).collect();
    assert_eq!(model.encode_with_dropout(bytes, 1.0, None), raw);
    assert_eq!(model.encode_with_dropout(bytes, 3.0, None), raw);
}

#[test]
fn dropout_is_seeded_lossless_and_applies_per_segment() {
    let model = trained_model(800);
    let text = pseudo_text(4, 20_000);
    let bytes = text.as_bytes();
    let greedy = model.encode(bytes).len() as f64;

    let a = model.encode_with_dropout(bytes, 0.3, Some(7));
    assert_eq!(a, model.encode_with_dropout(bytes, 0.3, Some(7)));
    assert_ne!(a, model.encode_with_dropout(bytes, 0.3, Some(8)));
    assert_eq!(model.decode(&a).unwrap(), bytes);

    // Whole-document dropout used to give ~+0% tokens; per-segment dropout must fragment.
    let p5 = model.encode_with_dropout(bytes, 0.5, Some(1)).len() as f64;
    let p1 = model.encode_with_dropout(bytes, 0.1, Some(1)).len() as f64;
    assert!(p5 > 1.3 * greedy, "p=0.5 gave {p5} tokens vs {greedy} greedy");
    assert!(p5 > p1 && p1 > greedy, "fertility must grow with p: {greedy} {p1} {p5}");
}

#[test]
fn super_bpe_dropout_fragments_whole_text_models() {
    // Review finding: with one document-wide skip list, every skipped merge came back after the
    // next merge anywhere in the document, so p = 0.5 gave +0% tokens on SuperBPE models.
    let model = super_bpe_model();
    for n in [200, 2000, 20_000] {
        let text = pseudo_text(9, n);
        let bytes = text.as_bytes();
        let greedy = model.encode(bytes);
        assert_eq!(model.encode_with_dropout(bytes, 0.0, Some(1)), greedy);
        let a = model.encode_with_dropout(bytes, 0.5, Some(3));
        assert_eq!(a, model.encode_with_dropout(bytes, 0.5, Some(3)));
        assert_eq!(model.decode(&a).unwrap(), bytes);
        let mean = (0..5).map(|s| model.encode_with_dropout(bytes, 0.5, Some(s)).len()).sum::<usize>() as f64 / 5.0;
        assert!(mean > 1.3 * greedy.len() as f64, "{n} B: p=0.5 gave {mean} tokens vs {} greedy", greedy.len());
    }
}

#[test]
fn single_word_dropout_follows_hf_semantics() {
    // One unit with merges (a,b) then (ab,c): with p in (0,1) every outcome is one of the three
    // prefixes of the merge chain, and all three appear.
    let corpus = vec![(b"abc".to_vec(), 10u64)];
    let model = BottomUpTrainer::new(&BpeScorer, 258).train(&corpus).unwrap();
    let (ab, abc) = (model.merges[0].2, model.merges[1].2);
    let (a, b, c) = (b'a' as TokenId, b'b' as TokenId, b'c' as TokenId);
    let mut seen = std::collections::HashSet::new();
    for seed in 0..200 {
        let ids = model.encode_with_dropout(b"abc", 0.5, Some(seed));
        assert!(ids == vec![abc] || ids == vec![ab, c] || ids == vec![a, b, c], "unexpected {ids:?}");
        seen.insert(ids);
    }
    assert_eq!(seen.len(), 3, "{seen:?}");
}

#[test]
fn special_tokens_are_matched_before_bpe() {
    let mut model = trained_model(800);
    let eot = model.vocab.add_special_token("<|endoftext|>").unwrap();
    let pad = model.vocab.add_special_token("<pad>").unwrap();
    let text = "hello world<|endoftext|>world<pad><pad>tail <|endoftext";
    let ids = model.encode(text.as_bytes());

    let mut expected = Vec::new();
    expected.extend(model.encode(b"hello world"));
    expected.push(eot);
    expected.extend(model.encode(b"world"));
    expected.extend([pad, pad]);
    expected.extend(model.encode(b"tail <|endoftext"));
    assert_eq!(ids, expected);
    assert_eq!(model.decode_lossless(&ids).unwrap(), text);

    // Specials survive dropout; everything else falls back to bytes at p = 1.
    let dropped = model.encode_with_dropout(b"ab<pad>c", 1.0, None);
    assert_eq!(dropped, vec![b'a' as TokenId, b'b' as TokenId, pad, b'c' as TokenId]);
}

#[test]
fn decode_rejects_unknown_ids() {
    let model = trained_model(300);
    assert!(model.decode(&[104, 999_999, 105]).is_err());
    assert!(model.decode_lossless(&[104, 999_999]).is_err());
    assert_eq!(model.decode(&[104, 105]).unwrap(), b"hi");
}
