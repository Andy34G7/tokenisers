use std::collections::BinaryHeap;
use ahash::AHashMap;
use rayon::prelude::*;

use crate::arena::{SymbolArena, NULL_NODE};
use crate::error::{Result, TokeniserError};
use crate::index::{InvertedIndex, OccurrenceRef, Pair};
use crate::scoring::{PairContext, PairScorer, ScoredPair};
use crate::vocab::{TokenId, Vocabulary, BASE_VOCAB_SIZE};

/// Trained Bottom-Up Model containing vocabulary and ordered merge operations.
#[derive(Clone, Debug)]
pub struct BottomUpModel {
    pub vocab: Vocabulary,
    pub merges: Vec<(TokenId, TokenId, TokenId)>,
    pub merge_ranks: AHashMap<Pair, (TokenId, u32)>,
}

impl BottomUpModel {
    pub fn new(
        vocab: Vocabulary,
        merges: Vec<(TokenId, TokenId, TokenId)>,
    ) -> Self {
        let mut merge_ranks = AHashMap::with_capacity(merges.len());
        for (rank, &(u, v, merged)) in merges.iter().enumerate() {
            merge_ranks.insert(Pair::new(u, v), (merged, rank as u32));
        }
        Self {
            vocab,
            merges,
            merge_ranks,
        }
    }

    /// Deterministic greedy encoding of a byte slice into token IDs.
    pub fn encode(&self, bytes: &[u8]) -> Vec<TokenId> {
        if bytes.is_empty() {
            return Vec::new();
        }

        // Initialize with base byte tokens
        let mut symbols: Vec<TokenId> = bytes.iter().map(|&b| b as TokenId).collect();
        if symbols.len() < 2 {
            return symbols;
        }

        loop {
            let mut best_pair: Option<(usize, Pair, TokenId, u32)> = None;

            for i in 0..(symbols.len() - 1) {
                let pair = Pair::new(symbols[i], symbols[i + 1]);
                if let Some(&(merged_id, rank)) = self.merge_ranks.get(&pair) {
                    match best_pair {
                        None => {
                            best_pair = Some((i, pair, merged_id, rank));
                        }
                        Some((_, _, _, best_rank)) if rank < best_rank => {
                            best_pair = Some((i, pair, merged_id, rank));
                        }
                        _ => {}
                    }
                }
            }

            match best_pair {
                None => break,
                Some((idx, _, merged_id, _)) => {
                    symbols[idx] = merged_id;
                    symbols.remove(idx + 1);
                }
            }
        }

        symbols
    }

    /// Parallel batch encoding across multiple texts.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<TokenId>> {
        texts
            .par_iter()
            .map(|text| self.encode(text.as_bytes()))
            .collect()
    }

    /// Decodes token IDs back to raw bytes.
    pub fn decode(&self, ids: &[TokenId]) -> Vec<u8> {
        self.vocab.decode(ids)
    }

    /// Decodes token IDs back to a UTF-8 String (byte lossless).
    pub fn decode_lossless(&self, ids: &[TokenId]) -> Result<String> {
        self.vocab.decode_lossless(ids)
    }
}

/// Pre-tokenizes text lines in parallel into word frequency counts using Rayon.
pub fn parallel_count_words(lines: &[&str]) -> AHashMap<Vec<u8>, u64> {
    lines
        .par_chunks(2048)
        .map(|chunk| {
            let mut local_counts: AHashMap<Vec<u8>, u64> = AHashMap::default();
            for line in chunk {
                // Word splitting on whitespace preserving content
                for word in line.split_whitespace() {
                    *local_counts.entry(word.as_bytes().to_vec()).or_insert(0) += 1;
                }
            }
            local_counts
        })
        .reduce(AHashMap::default, |mut map1, map2| {
            for (word, count) in map2 {
                *map1.entry(word).or_insert(0) += count;
            }
            map1
        })
}

/// Unified Bottom-Up training engine supporting arbitrary PairScorers, O(1) splicing,
/// and version-tombstoned priority queues.
pub struct BottomUpTrainer<'a> {
    pub scorer: &'a dyn PairScorer,
    pub target_vocab_size: usize,
    pub min_frequency: u64,
}

impl<'a> BottomUpTrainer<'a> {
    pub fn new(scorer: &'a dyn PairScorer, target_vocab_size: usize) -> Self {
        Self {
            scorer,
            target_vocab_size,
            min_frequency: 1,
        }
    }

    pub fn with_min_frequency(mut self, min_frequency: u64) -> Self {
        self.min_frequency = min_frequency;
        self
    }

    /// Trains a Bottom-Up tokenizer from unique words with their corpus frequencies.
    pub fn train(&self, word_counts: &[(Vec<u8>, u64)]) -> Result<BottomUpModel> {
        if self.target_vocab_size < BASE_VOCAB_SIZE {
            return Err(TokeniserError::VocabSizeTooSmall(self.target_vocab_size));
        }
        if word_counts.is_empty() {
            return Err(TokeniserError::EmptyCorpus);
        }

        let mut vocab = Vocabulary::new_byte_level();
        let mut arena = SymbolArena::with_capacity(word_counts.len() * 8, word_counts.len());

        let mut token_counts: AHashMap<TokenId, u64> = AHashMap::default();
        let mut total_tokens = 0u64;

        // 1. Populate arena with words
        for (word_bytes, freq) in word_counts {
            if word_bytes.is_empty() || *freq == 0 {
                continue;
            }
            let tokens: Vec<TokenId> = word_bytes.iter().map(|&b| b as TokenId).collect();
            for &tok in &tokens {
                *token_counts.entry(tok).or_insert(0) += *freq;
                total_tokens += *freq;
            }
            arena.add_word(&tokens, *freq);
        }

        // 2. Build initial pair counts and inverted index
        let mut pair_counts: AHashMap<Pair, u64> = AHashMap::default();
        let mut inverted_index = InvertedIndex::with_capacity(arena.word_count() * 4);
        let mut pair_versions: AHashMap<Pair, u32> = AHashMap::default();

        for word_id in 0..arena.word_count() as u32 {
            let word = arena.get_word(word_id).unwrap();
            let mut curr = word.head;
            while curr != NULL_NODE {
                let node = arena.get_node(curr).unwrap();
                let next = node.next;
                if next != NULL_NODE {
                    let next_node = arena.get_node(next).unwrap();
                    let pair = Pair::new(node.token, next_node.token);
                    *pair_counts.entry(pair).or_insert(0) += word.freq;
                    inverted_index.add_occurrence(pair, OccurrenceRef::new(word_id, curr, node.version));
                }
                curr = next;
            }
        }

        // 3. Build priority queue
        let mut heap = BinaryHeap::with_capacity(pair_counts.len());
        for (&pair, &count) in &pair_counts {
            if count < self.min_frequency {
                continue;
            }
            let len_u = vocab.get_bytes(pair.0).map_or(1, |b| b.len());
            let len_v = vocab.get_bytes(pair.1).map_or(1, |b| b.len());

            let ctx = PairContext {
                freq_uv: count,
                freq_u: token_counts.get(&pair.0).copied().unwrap_or(0),
                freq_v: token_counts.get(&pair.1).copied().unwrap_or(0),
                total_tokens,
                active_vocab_size: vocab.len(),
                len_u,
                len_v,
            };

            let score = self.scorer.score(pair, &ctx);
            if score.is_finite() {
                heap.push(ScoredPair {
                    score,
                    pair,
                    version: 0,
                });
            }
        }

        let mut merges = Vec::new();

        // 4. Merge loop
        while vocab.len() < self.target_vocab_size {
            // Find next valid candidate from heap
            let mut next_merge: Option<Pair> = None;

            while let Some(candidate) = heap.pop() {
                let current_version = pair_versions.get(&candidate.pair).copied().unwrap_or(0);
                let current_count = pair_counts.get(&candidate.pair).copied().unwrap_or(0);

                if candidate.version == current_version && current_count >= self.min_frequency {
                    next_merge = Some(candidate.pair);
                    break;
                }
                // Discard stale tombstone
            }

            let pair = match next_merge {
                Some(p) => p,
                None => break, // No more valid pairs
            };

            // Register merged token
            let bytes_u = vocab.get_bytes(pair.0).unwrap().to_vec();
            let bytes_v = vocab.get_bytes(pair.1).unwrap().to_vec();
            let mut merged_bytes = Vec::with_capacity(bytes_u.len() + bytes_v.len());
            merged_bytes.extend_from_slice(&bytes_u);
            merged_bytes.extend_from_slice(&bytes_v);

            let new_token = vocab.add_token(merged_bytes);
            merges.push((pair.0, pair.1, new_token));

            let occurrences = inverted_index.take(&pair).unwrap_or_default();

            // Mark this pair's count as consumed
            pair_counts.remove(&pair);
            let p_ver = pair_versions.entry(pair).or_insert(0);
            *p_ver = p_ver.wrapping_add(1);

            let mut affected_pairs: AHashMap<Pair, ()> = AHashMap::default();

            for occ in occurrences {
                if !occ.is_valid(pair, &arena) {
                    continue; // Skip stale occurrence
                }

                let left_idx = occ.node_idx;
                let right_idx = arena.get_node(left_idx).unwrap().next;
                let word_id = occ.word_id;
                let word_freq = arena.get_word(word_id).unwrap().freq;

                let left_node = *arena.get_node(left_idx).unwrap();
                let right_node = *arena.get_node(right_idx).unwrap();

                let prev_idx = left_node.prev;
                let next_idx = right_node.next;

                // Decrement neighbor pair counts
                if prev_idx != NULL_NODE {
                    let prev_node = arena.get_node(prev_idx).unwrap();
                    let old_prev_pair = Pair::new(prev_node.token, left_node.token);
                    if let Some(cnt) = pair_counts.get_mut(&old_prev_pair) {
                        *cnt = cnt.saturating_sub(word_freq);
                    }
                    affected_pairs.insert(old_prev_pair, ());
                }

                if next_idx != NULL_NODE {
                    let next_node = arena.get_node(next_idx).unwrap();
                    let old_next_pair = Pair::new(right_node.token, next_node.token);
                    if let Some(cnt) = pair_counts.get_mut(&old_next_pair) {
                        *cnt = cnt.saturating_sub(word_freq);
                    }
                    affected_pairs.insert(old_next_pair, ());
                }

                // Splicing
                let _ = arena.splice_pair(left_idx, right_idx, new_token, word_id);

                // Increment new neighbor pair counts and record occurrences
                if prev_idx != NULL_NODE {
                    let prev_node = arena.get_node(prev_idx).unwrap();
                    let new_prev_pair = Pair::new(prev_node.token, new_token);
                    *pair_counts.entry(new_prev_pair).or_insert(0) += word_freq;
                    inverted_index.add_occurrence(
                        new_prev_pair,
                        OccurrenceRef::new(word_id, prev_idx, prev_node.version),
                    );
                    affected_pairs.insert(new_prev_pair, ());
                }

                if next_idx != NULL_NODE {
                    let next_node = arena.get_node(next_idx).unwrap();
                    let new_next_pair = Pair::new(new_token, next_node.token);
                    *pair_counts.entry(new_next_pair).or_insert(0) += word_freq;
                    let curr_ver = arena.get_node(left_idx).unwrap().version;
                    inverted_index.add_occurrence(
                        new_next_pair,
                        OccurrenceRef::new(word_id, left_idx, curr_ver),
                    );
                    affected_pairs.insert(new_next_pair, ());
                }

                // Update token stats
                if let Some(cnt) = token_counts.get_mut(&pair.0) {
                    *cnt = cnt.saturating_sub(word_freq);
                }
                if let Some(cnt) = token_counts.get_mut(&pair.1) {
                    *cnt = cnt.saturating_sub(word_freq);
                }
                *token_counts.entry(new_token).or_insert(0) += word_freq;
                total_tokens = total_tokens.saturating_sub(word_freq);
            }

            // Push updated scores for affected pairs into heap
            for (aff_pair, _) in affected_pairs {
                let count = pair_counts.get(&aff_pair).copied().unwrap_or(0);
                let ver = pair_versions.entry(aff_pair).or_insert(0);
                *ver = ver.wrapping_add(1);
                let current_ver = *ver;

                if count >= self.min_frequency {
                    let len_u = vocab.get_bytes(aff_pair.0).map_or(1, |b| b.len());
                    let len_v = vocab.get_bytes(aff_pair.1).map_or(1, |b| b.len());

                    let ctx = PairContext {
                        freq_uv: count,
                        freq_u: token_counts.get(&aff_pair.0).copied().unwrap_or(0),
                        freq_v: token_counts.get(&aff_pair.1).copied().unwrap_or(0),
                        total_tokens,
                        active_vocab_size: vocab.len(),
                        len_u,
                        len_v,
                    };

                    let score = self.scorer.score(aff_pair, &ctx);
                    if score.is_finite() {
                        heap.push(ScoredPair {
                            score,
                            pair: aff_pair,
                            version: current_ver,
                        });
                    }
                }
            }
        }

        Ok(BottomUpModel::new(vocab, merges))
    }
}
