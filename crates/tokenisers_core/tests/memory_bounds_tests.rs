//! Peak heap allocation bounds for Unigram k-best and FFBS sampling and Unigram training.
//!
//! A counting global allocator records the peak number of live heap bytes, so these tests catch
//! O(text length * k) or per-distinct-substring memory without needing a big input.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard};

use tokenisers_core::engine::unigram::{CompactTrie, UnigramTrainer, ViterbiDecoder};
use tokenisers_core::inference::unigram_sample::UnigramSampler;

struct CountingAlloc;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            PEAK.fetch_max(LIVE.fetch_add(layout.size(), Relaxed) + layout.size(), Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            let live = LIVE.fetch_add(new_size, Relaxed) + new_size;
            PEAK.fetch_max(live, Relaxed);
            LIVE.fetch_sub(layout.size(), Relaxed);
        }
        p
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Held for a whole test body: tests run one at a time, so no other test's setup allocations
/// (input text, a 2.6 MB trie) land in a measured peak.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Peak extra heap bytes while `f` runs. Call it while holding [`serial`].
fn peak_bytes<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let out = f();
    (out, PEAK.load(Relaxed).saturating_sub(base))
}

const MB: usize = 1 << 20;

/// All 256 bytes plus every 2..=4-gram over "abcd": many segmentations at every position.
fn ngram_vocab() -> Vec<(Vec<u8>, u32, f64)> {
    let mut vocab: Vec<(Vec<u8>, u32, f64)> = (0..=255u8).map(|b| (vec![b], b as u32, -6.0)).collect();
    let mut grams: Vec<Vec<u8>> = vec![Vec::new()];
    for _ in 0..4 {
        grams = grams
            .iter()
            .flat_map(|g| b"abcd".iter().map(move |&c| [g.as_slice(), &[c]].concat()))
            .collect();
        for g in grams.iter().filter(|g| g.len() >= 2) {
            let id = vocab.len() as u32;
            vocab.push((g.clone(), id, -3.0 - g.len() as f64 * 0.7 - (id % 7) as f64 * 0.1));
        }
    }
    vocab
}

/// Space-separated words of 3..=12 letters over "abcd", about `n` bytes.
fn words(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut text = Vec::with_capacity(n + 16);
    while text.len() < n {
        text.extend((0..rng.usize(3..=12)).map(|_| b"abcd"[rng.usize(..4)]));
        text.push(b' ');
    }
    text
}

#[test]
fn kbest_memory_does_not_grow_with_document_length() {
    let _serial = serial();
    let trie = CompactTrie::build(&ngram_vocab());
    let sampler = UnigramSampler::new(&trie);
    let text = words(40_000, 1);
    let (tokens, peak) = peak_bytes(|| sampler.sample_kbest(&text, 64, 1.0, Some(3)));
    assert!(!tokens.is_empty());
    // A whole-text table would be 40 000 positions x 64 paths x 32 bytes = 80 MB.
    assert!(peak < 2 * MB, "k-best on 40 KB of words peaked at {} KB", peak / 1024);
}

#[test]
fn ffbs_memory_does_not_grow_with_document_length() {
    let _serial = serial();
    let vocab = ngram_vocab();
    let trie = CompactTrie::build(&vocab);
    let sampler = UnigramSampler::new(&trie);
    let text = words(100_000, 7);
    let (tokens, peak) = peak_bytes(|| sampler.sample_ffbs(&text, 1.0, Some(8)));
    let decoded: Vec<u8> = tokens.iter().flat_map(|&t| vocab[t as usize].0.clone()).collect();
    assert_eq!(decoded, text);
    // A whole-text lattice and alpha table is ~160 bytes per byte: about 16 MB here.
    assert!(peak < 2 * MB, "FFBS on 100 KB of words peaked at {} KB", peak / 1024);
}

#[test]
fn kbest_memory_is_bounded_for_huge_k_and_long_segments() {
    let _serial = serial();
    let vocab = ngram_vocab();
    let trie = CompactTrie::build(&vocab);
    let sampler = UnigramSampler::new(&trie);

    // A huge nbest_size on ordinary text: each word's table holds at most its own paths.
    let text = words(2_000, 2);
    let (_, peak) = peak_bytes(|| sampler.sample_kbest(&text, usize::MAX, 1.0, Some(4)));
    assert!(peak < 16 * MB, "k = usize::MAX on 2 KB of words peaked at {} MB", peak / MB);

    // One 20 KB segment (no whitespace): k is capped so the table stays within the budget.
    let long: Vec<u8> = words(20_000, 5).into_iter().filter(|&b| b != b' ').collect();
    for k in [1024, usize::MAX] {
        let (tokens, peak) = peak_bytes(|| sampler.sample_kbest(&long, k, 1.0, Some(6)));
        let decoded: Vec<u8> = tokens.iter().flat_map(|&t| vocab[t as usize].0.clone()).collect();
        assert_eq!(decoded, long, "k-best must stay lossless when k is capped");
        assert!(peak < 64 * MB, "k = {k} on a 20 KB segment peaked at {} MB", peak / MB);
    }
}

#[test]
fn sampling_one_long_segment_needs_no_per_byte_lattice_or_k_cap() {
    let _serial = serial();
    let vocab = ngram_vocab();
    let trie = CompactTrie::build(&vocab);
    let sampler = UnigramSampler::new(&trie);
    let decode = |tokens: &[u32]| -> Vec<u8> {
        tokens.iter().flat_map(|&t| vocab[t as usize].0.clone()).collect()
    };
    // 600 KB without whitespace is one segment (an unspaced CJK book, minified JSON, base64).
    // A whole-segment lattice is ~150 B per byte, and a table budget of 2^20 entries / (len + 1)
    // would cut k to 1 here, so `nbest_size = 64` would silently return the Viterbi path.
    let long: Vec<u8> = words(700_000, 13).into_iter().filter(|&b| b != b' ').collect();
    let n = long.len();
    assert!(n > 520_000, "need a segment longer than 2^20 / 2 bytes, got {n}");
    let viterbi = ViterbiDecoder::decode(&long, &trie, None).unwrap().1;

    let (tokens, peak) = peak_bytes(|| sampler.sample_kbest(&long, 64, 1.0, Some(0)));
    // The output ids (4 B per byte reserved) plus k-best rows of about 2 * sqrt(len * 4) positions.
    assert!(peak < 8 * n + 8 * MB, "k-best on a {n}-byte segment peaked at {} MB", peak / MB);
    assert_eq!(decode(&tokens), long);
    let differs = tokens != viterbi
        || (1..3).any(|seed| sampler.sample_kbest(&long, 64, 1.0, Some(seed)) != viterbi);
    assert!(differs, "k-best with k = 64 on a long segment fell back to Viterbi");

    let (tokens, peak) = peak_bytes(|| sampler.sample_ffbs(&long, 1.0, Some(5)));
    assert_eq!(decode(&tokens), long);
    // An 8-byte forward table per byte, the output ids and one block of the lattice.
    assert!(peak < 16 * n + 2 * MB, "FFBS on a {n}-byte segment peaked at {} MB", peak / MB);
}

#[test]
fn unigram_training_memory_is_not_per_distinct_substring() {
    let _serial = serial();
    // Distinct random words: nearly every 2..=16-byte substring is distinct, the worst case for
    // counting substrings in a hash map (about 100 MB here, 370 B per corpus byte). The
    // fixed allowance covers the seed-vocabulary trie and EM buffers, which do not grow with
    // the corpus.
    let mut rng = fastrand::Rng::with_seed(9);
    let corpus: Vec<(Vec<u8>, u64)> = (0..30_000)
        .map(|_| ((0..rng.usize(5..=15)).map(|_| rng.u8(b'a'..=b'z')).collect(), rng.u64(1..20)))
        .collect();
    let corpus_bytes: usize = corpus.iter().map(|(w, _)| w.len()).sum();
    let (model, peak) = peak_bytes(|| UnigramTrainer::new(300, 0.15).train(&corpus).unwrap());
    assert_eq!(model.vocab.len(), 300);
    assert!(
        peak < 16 * MB + 50 * corpus_bytes,
        "training on {corpus_bytes} bytes peaked at {} MB ({} B/byte)",
        peak / MB,
        peak / corpus_bytes
    );
}
