use std::collections::BinaryHeap;
use ahash::AHashMap;

use crate::arena::{SymbolArena, NULL_NODE};
use crate::engine::bottom_up::BottomUpModel;
use crate::error::{Result, TokeniserError};
use crate::index::{InvertedIndex, OccurrenceRef, Pair};
use crate::scoring::{PairContext, PairScorer, ScoredPair};
use crate::vocab::{TokenId, Vocabulary, BASE_VOCAB_SIZE};

/// Configuration for the Two-Stage SuperBPE curriculum.
#[derive(Debug, Clone)]
pub struct SuperBpeConfig {
    pub v_switch: usize,
    pub v_final: usize,
}

impl SuperBpeConfig {
    pub fn new(v_switch: usize, v_final: usize) -> Result<Self> {
        if v_switch < BASE_VOCAB_SIZE {
            return Err(TokeniserError::VocabSizeTooSmall(v_switch));
        }
        if v_switch > v_final {
            return Err(TokeniserError::InvalidConfiguration(format!(
                "v_switch ({v_switch}) cannot exceed v_final ({v_final})"
            )));
        }
        Ok(Self { v_switch, v_final })
    }
}

/// Boundary junction between two segments in the arena.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryJunction {
    pub left_word_idx: u32,
    pub right_word_idx: u32,
    pub freq: u64,
}

/// Two-Stage SuperBPE training engine.
pub struct SuperBpeTrainer<'a> {
    pub config: SuperBpeConfig,
    pub scorer: &'a dyn PairScorer,
    pub min_frequency: u64,
}

impl<'a> SuperBpeTrainer<'a> {
    pub fn new(config: SuperBpeConfig, scorer: &'a dyn PairScorer) -> Self {
        Self {
            config,
            scorer,
            min_frequency: 1,
        }
    }

    /// Splits text into alternating words and whitespace segments.
    pub fn segment_text(text: &str) -> Vec<Vec<u8>> {
        let mut segments = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let start = i;
            let is_ws = bytes[i].is_ascii_whitespace();
            while i < bytes.len() && bytes[i].is_ascii_whitespace() == is_ws {
                i += 1;
            }
            segments.push(bytes[start..i].to_vec());
        }
        segments
    }

    /// Relinks boundary junctions across words and populates heap with cross-word candidate pairs.
    #[allow(clippy::too_many_arguments)]
    fn relink_boundaries(
        &self,
        arena: &mut SymbolArena,
        junctions: &[BoundaryJunction],
        inverted_index: &mut InvertedIndex,
        pair_counts: &mut AHashMap<Pair, u64>,
        pair_versions: &mut AHashMap<Pair, u32>,
        heap: &mut BinaryHeap<ScoredPair>,
        vocab: &Vocabulary,
        token_counts: &AHashMap<TokenId, u64>,
        total_tokens: u64,
    ) {
        for junction in junctions {
            let left = match arena.get_word(junction.left_word_idx) {
                Some(w) => w.tail,
                None => continue,
            };
            let right = match arena.get_word(junction.right_word_idx) {
                Some(w) => w.head,
                None => continue,
            };

            // Resolve currently active nodes
            if left == NULL_NODE
                || right == NULL_NODE
                || !arena.is_node_active(left)
                || !arena.is_node_active(right)
            {
                continue;
            }

            // Link nodes in arena
            arena.link_nodes(left, right);

            let tok_u = arena.get_token(left);
            let tok_v = arena.get_token(right);
            let pair = Pair::new(tok_u, tok_v);

            *pair_counts.entry(pair).or_insert(0) += junction.freq;
            let ver = arena.get_node(left).unwrap().version;
            inverted_index.add_occurrence(
                pair,
                OccurrenceRef::new(junction.left_word_idx, left, ver),
            );

            // Update version and push to heap
            let p_ver = pair_versions.entry(pair).or_insert(0);
            *p_ver = p_ver.wrapping_add(1);

            let len_u = vocab.get_bytes(tok_u).map_or(1, |b| b.len());
            let len_v = vocab.get_bytes(tok_v).map_or(1, |b| b.len());
            let ctx = PairContext {
                freq_uv: pair_counts[&pair],
                freq_u: token_counts.get(&tok_u).copied().unwrap_or(0),
                freq_v: token_counts.get(&tok_v).copied().unwrap_or(0),
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
                    version: *p_ver,
                });
            }
        }
    }

    /// Trains Two-Stage SuperBPE on a collection of document lines.
    pub fn train(&self, lines: &[&str]) -> Result<BottomUpModel> {
        if lines.is_empty() {
            return Err(TokeniserError::EmptyCorpus);
        }

        let mut vocab = Vocabulary::new_byte_level();
        let mut arena = SymbolArena::new();
        let mut junctions = Vec::new();

        let mut token_counts: AHashMap<TokenId, u64> = AHashMap::default();
        let mut total_tokens = 0u64;

        // Group sequences into segments
        for line in lines {
            if line.is_empty() {
                continue;
            }
            let segments = Self::segment_text(line);
            let mut prev_word_idx: Option<u32> = None;

            for seg in segments {
                if seg.is_empty() {
                    continue;
                }
                let tokens: Vec<TokenId> = seg.iter().map(|&b| b as TokenId).collect();
                for &tok in &tokens {
                    *token_counts.entry(tok).or_insert(0) += 1;
                    total_tokens += 1;
                }
                let word_id = arena.add_word(&tokens, 1);

                if let Some(prev) = prev_word_idx {
                    junctions.push(BoundaryJunction {
                        left_word_idx: prev,
                        right_word_idx: word_id,
                        freq: 1,
                    });
                }
                prev_word_idx = Some(word_id);
            }
        }

        // Build initial pair counts and inverted index (Stage 1: intra-segment only)
        let mut pair_counts: AHashMap<Pair, u64> = AHashMap::default();
        let mut inverted_index = InvertedIndex::new();
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

        // Build priority queue
        let mut heap = BinaryHeap::new();
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
        let mut stage = 1;

        // Merge loop across Stage 1 and Stage 2
        while vocab.len() < self.config.v_final {
            // Check if stage transition is triggered
            if stage == 1 && vocab.len() >= self.config.v_switch {
                log::info!("SuperBPE: lifting whitespace boundaries at v_switch = {}", self.config.v_switch);
                stage = 2;
                self.relink_boundaries(
                    &mut arena,
                    &junctions,
                    &mut inverted_index,
                    &mut pair_counts,
                    &mut pair_versions,
                    &mut heap,
                    &vocab,
                    &token_counts,
                    total_tokens,
                );
            }

            // Pop next valid pair
            let mut next_merge: Option<Pair> = None;
            while let Some(candidate) = heap.pop() {
                let current_version = pair_versions.get(&candidate.pair).copied().unwrap_or(0);
                let current_count = pair_counts.get(&candidate.pair).copied().unwrap_or(0);

                if candidate.version == current_version && current_count >= self.min_frequency {
                    next_merge = Some(candidate.pair);
                    break;
                }
            }

            let pair = match next_merge {
                Some(p) => p,
                None => {
                    if stage == 1 && vocab.len() < self.config.v_final {
                        log::info!(
                            "SuperBPE: early transition to stage 2 (intra-word pairs exhausted at vocab len {})",
                            vocab.len()
                        );
                        stage = 2;
                        self.relink_boundaries(
                            &mut arena,
                            &junctions,
                            &mut inverted_index,
                            &mut pair_counts,
                            &mut pair_versions,
                            &mut heap,
                            &vocab,
                            &token_counts,
                            total_tokens,
                        );
                        continue;
                    }
                    break;
                }
            };

            let bytes_u = vocab.get_bytes(pair.0).unwrap().to_vec();
            let bytes_v = vocab.get_bytes(pair.1).unwrap().to_vec();
            let mut merged_bytes = Vec::with_capacity(bytes_u.len() + bytes_v.len());
            merged_bytes.extend_from_slice(&bytes_u);
            merged_bytes.extend_from_slice(&bytes_v);

            let new_token = vocab.add_token(merged_bytes);
            merges.push((pair.0, pair.1, new_token));

            let occurrences = inverted_index.take(&pair).unwrap_or_default();

            pair_counts.remove(&pair);
            let p_ver = pair_versions.entry(pair).or_insert(0);
            *p_ver = p_ver.wrapping_add(1);

            let mut affected_pairs: AHashMap<Pair, ()> = AHashMap::default();

            for occ in occurrences {
                if !occ.is_valid(pair, &arena) {
                    continue;
                }

                let left_idx = occ.node_idx;
                let right_idx = arena.get_node(left_idx).unwrap().next;
                let word_id = occ.word_id;
                let word_freq = arena.get_word(word_id).map_or(1, |w| w.freq);

                let left_node = *arena.get_node(left_idx).unwrap();
                let right_node = *arena.get_node(right_idx).unwrap();

                let prev_idx = left_node.prev;
                let next_idx = right_node.next;

                if prev_idx != NULL_NODE {
                    let prev_node = arena.get_node(prev_idx).unwrap();
                    let old_prev = Pair::new(prev_node.token, left_node.token);
                    if let Some(cnt) = pair_counts.get_mut(&old_prev) {
                        *cnt = cnt.saturating_sub(word_freq);
                    }
                    affected_pairs.insert(old_prev, ());
                }

                if next_idx != NULL_NODE {
                    let next_node = arena.get_node(next_idx).unwrap();
                    let old_next = Pair::new(right_node.token, next_node.token);
                    if let Some(cnt) = pair_counts.get_mut(&old_next) {
                        *cnt = cnt.saturating_sub(word_freq);
                    }
                    affected_pairs.insert(old_next, ());
                }

                let _ = arena.splice_pair(left_idx, right_idx, new_token, word_id);

                if prev_idx != NULL_NODE {
                    let prev_node = arena.get_node(prev_idx).unwrap();
                    let new_prev = Pair::new(prev_node.token, new_token);
                    *pair_counts.entry(new_prev).or_insert(0) += word_freq;
                    inverted_index.add_occurrence(
                        new_prev,
                        OccurrenceRef::new(word_id, prev_idx, prev_node.version),
                    );
                    affected_pairs.insert(new_prev, ());
                }

                if next_idx != NULL_NODE {
                    let next_node = arena.get_node(next_idx).unwrap();
                    let new_next = Pair::new(new_token, next_node.token);
                    *pair_counts.entry(new_next).or_insert(0) += word_freq;
                    let curr_ver = arena.get_node(left_idx).unwrap().version;
                    inverted_index.add_occurrence(
                        new_next,
                        OccurrenceRef::new(word_id, left_idx, curr_ver),
                    );
                    affected_pairs.insert(new_next, ());
                }

                if let Some(cnt) = token_counts.get_mut(&pair.0) {
                    *cnt = cnt.saturating_sub(word_freq);
                }
                if let Some(cnt) = token_counts.get_mut(&pair.1) {
                    *cnt = cnt.saturating_sub(word_freq);
                }
                *token_counts.entry(new_token).or_insert(0) += word_freq;
                total_tokens = total_tokens.saturating_sub(word_freq);
            }

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
