//! Brute-force references shared by integration tests.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};

use tokenisers_core::curricula::morpheme::PairMorphologyStats;
use tokenisers_core::index::Pair;
use tokenisers_core::scoring::{PairContext, PairScorer, ScoredPair};
use tokenisers_core::vocab::{TokenId, Vocabulary};

pub type Merges = Vec<(TokenId, TokenId, TokenId)>;

/// Replaces every (u, v) in `seq` by `w`, scanning left to right (overlaps resolve leftmost-first).
pub fn apply_leftmost(seq: &[TokenId], u: TokenId, v: TokenId, w: TokenId) -> Vec<TokenId> {
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == u && seq[i + 1] == v {
            out.push(w);
            i += 2;
        } else {
            out.push(seq[i]);
            i += 1;
        }
    }
    out
}

/// Reference trainer: every step recounts pair counts, token counts and N from scratch, scores
/// every pair, merges the maximum under `ScoredPair` ordering and re-segments every word.
/// If the scorer annotates any word (`word_boundaries`), it also recounts each pair's intra /
/// cross-boundary occurrences from the byte spans of the current segmentation and scores with
/// `score_morph`. Returns the merges and the final segmentation of each non-empty word (input order).
pub fn reference_train(
    scorer: &dyn PairScorer,
    word_counts: &[(Vec<u8>, u64)],
    target_vocab_size: usize,
    min_frequency: u64,
) -> (Merges, Vec<Vec<TokenId>>) {
    let min_frequency = min_frequency.max(1);
    let mut vocab = Vocabulary::new_byte_level();
    let mut words: Vec<(Vec<TokenId>, u64)> = word_counts
        .iter()
        .filter(|(w, f)| !w.is_empty() && *f > 0)
        .map(|(w, f)| (w.iter().map(|&b| b as TokenId).collect(), *f))
        .collect();
    let bounds: Vec<Option<&[usize]>> = word_counts
        .iter()
        .filter(|(w, f)| !w.is_empty() && *f > 0)
        .map(|(w, _)| scorer.word_boundaries(w))
        .collect();
    let annotated = bounds.iter().any(Option::is_some);
    let mut merges = Vec::new();
    while vocab.len() < target_vocab_size {
        let mut pair_counts: BTreeMap<Pair, u64> = BTreeMap::new();
        let mut token_counts: HashMap<TokenId, u64> = HashMap::new();
        let mut total = 0u64;
        let mut morph: HashMap<Pair, PairMorphologyStats> = HashMap::new();
        for ((w, f), b) in words.iter().zip(&bounds) {
            let Some(b) = b else { continue };
            let mut start = 0;
            for pair in w.windows(2) {
                let len_u = vocab.get_bytes(pair[0]).unwrap().len();
                let end = start + len_u + vocab.get_bytes(pair[1]).unwrap().len();
                let stat = morph.entry(Pair::new(pair[0], pair[1])).or_default();
                if b.iter().any(|&x| start < x && x < end) {
                    stat.n_cross += f;
                } else {
                    stat.n_intra += f;
                }
                start += len_u;
            }
        }
        for (w, f) in &words {
            for t in w {
                *token_counts.entry(*t).or_default() += f;
                total += f;
            }
            for pair in w.windows(2) {
                *pair_counts.entry(Pair::new(pair[0], pair[1])).or_default() += f;
            }
        }
        let mut best: Option<ScoredPair> = None;
        for (&pair, &count) in &pair_counts {
            if count < min_frequency {
                continue;
            }
            let ctx = PairContext {
                freq_uv: count,
                freq_u: token_counts[&pair.0],
                freq_v: token_counts[&pair.1],
                total_tokens: total,
                active_vocab_size: vocab.len(),
                len_u: vocab.get_bytes(pair.0).unwrap().len(),
                len_v: vocab.get_bytes(pair.1).unwrap().len(),
            };
            let score = if annotated {
                scorer.score_morph(pair, &ctx, morph.get(&pair).copied().unwrap_or_default())
            } else {
                scorer.score(pair, &ctx)
            };
            if !score.is_finite() {
                continue;
            }
            let cand = ScoredPair { score, pair, version: 0 };
            if best.map_or(true, |b| cand > b) {
                best = Some(cand);
            }
        }
        let Some(best) = best else { break };
        let (u, v) = (best.pair.0, best.pair.1);
        let mut bytes = vocab.get_bytes(u).unwrap().to_vec();
        bytes.extend_from_slice(vocab.get_bytes(v).unwrap());
        let w = vocab.add_token(bytes);
        merges.push((u, v, w));
        for (seq, _) in words.iter_mut() {
            *seq = apply_leftmost(seq, u, v, w);
        }
    }
    (merges, words.into_iter().map(|(w, _)| w).collect())
}

/// Reference encoder: replays the merges in rank order over the whole input.
pub fn reference_encode(merges: &Merges, bytes: &[u8]) -> Vec<TokenId> {
    let mut seq: Vec<TokenId> = bytes.iter().map(|&b| b as TokenId).collect();
    for &(u, v, w) in merges {
        seq = apply_leftmost(&seq, u, v, w);
    }
    seq
}

/// Random word counts over `alphabet` (deterministic for a given rng).
pub fn random_corpus(rng: &mut fastrand::Rng, alphabet: &[u8], n_words: usize, max_len: usize) -> Vec<(Vec<u8>, u64)> {
    let mut counts: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    for _ in 0..n_words {
        let len = rng.usize(1..=max_len);
        let word: Vec<u8> = (0..len).map(|_| alphabet[rng.usize(0..alphabet.len())]).collect();
        *counts.entry(word).or_default() += rng.u64(1..6);
    }
    counts.into_iter().collect()
}

/// Deterministic pseudo-text: syllable words separated by mixed whitespace and punctuation.
pub fn pseudo_text(seed: u64, n_bytes: usize) -> String {
    const SYL: &[&str] = &[
        "the", "an", "ing", "tion", "re", "con", "pre", "ed", "er", "ly", "qu", "ex", "ab", "st",
        "pro", "ment", "ous", "al", "in", "de", "un", "ate", "ize", "ity", "o", "a", "e", "i",
    ];
    const SEP: &[&str] = &[" ", " ", " ", " ", " ", ", ", ". ", "\n", "  ", "\t", "; ", "! "];
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut s = String::with_capacity(n_bytes + 32);
    while s.len() < n_bytes {
        for _ in 0..rng.usize(1..=4) {
            s.push_str(SYL[rng.usize(0..SYL.len())]);
        }
        s.push_str(SEP[rng.usize(0..SEP.len())]);
    }
    s.truncate(n_bytes);
    s
}
