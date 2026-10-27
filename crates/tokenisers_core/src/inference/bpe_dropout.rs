use ahash::AHashMap;
use fastrand::Rng;

use crate::index::Pair;
use crate::vocab::TokenId;

/// Stochastic BPE encoder implementing Provilkov merge dropout (p in [0, 1]).
#[derive(Clone, Debug)]
pub struct BpeDropoutEncoder {
    merge_ranks: AHashMap<Pair, (TokenId, u32)>,
}

impl BpeDropoutEncoder {
    pub fn new(merges: &[(TokenId, TokenId, TokenId)]) -> Self {
        let mut merge_ranks = AHashMap::with_capacity(merges.len());
        for (rank, &(u, v, merged)) in merges.iter().enumerate() {
            merge_ranks.insert(Pair::new(u, v), (merged, rank as u32));
        }
        Self { merge_ranks }
    }

    pub fn from_ranks(merge_ranks: AHashMap<Pair, (TokenId, u32)>) -> Self {
        Self { merge_ranks }
    }

    /// Encodes a raw byte slice into token IDs using BPE-Dropout.
    ///
    /// - `dropout_prob`: merge dropout probability p in [0.0, 1.0].
    /// - `seed`: optional seed for deterministic PRNG reproducible testing.
    pub fn encode(&self, bytes: &[u8], dropout_prob: f32, seed: Option<u64>) -> Vec<TokenId> {
        if bytes.is_empty() {
            return Vec::new();
        }

        let mut symbols: Vec<TokenId> = bytes.iter().map(|&b| b as TokenId).collect();
        if symbols.len() < 2 {
            return symbols;
        }

        // Fast paths
        if dropout_prob <= 0.0 {
            return self.encode_greedy(symbols);
        }
        if dropout_prob >= 1.0 {
            return symbols;
        }

        let mut rng = Rng::new();
        if let Some(s) = seed {
            rng.seed(s);
        }

        while symbols.len() >= 2 {
            // 1. Gather all eligible candidate merges
            let mut candidates: Vec<(usize, Pair, TokenId, u32)> = Vec::new();
            for i in 0..(symbols.len() - 1) {
                let pair = Pair::new(symbols[i], symbols[i + 1]);
                if let Some(&(merged_id, rank)) = self.merge_ranks.get(&pair) {
                    candidates.push((i, pair, merged_id, rank));
                }
            }

            if candidates.is_empty() {
                break;
            }

            // 2. Bernoulli sampling with probability (1 - p)
            let mut kept = Vec::new();
            for cand in &candidates {
                if rng.f32() >= dropout_prob {
                    kept.push(*cand);
                }
            }

            // 3. Selection: if all candidates are dropped, terminate merge loop
            let chosen = if let Some(min_cand) = kept.into_iter().min_by_key(|c| c.3) {
                min_cand
            } else {
                break;
            };

            let (idx, _, merged_id, _) = chosen;

            // 4. Splicing
            symbols[idx] = merged_id;
            symbols.remove(idx + 1);
        }

        symbols
    }

    /// Greedy deterministic BPE encoding.
    fn encode_greedy(&self, mut symbols: Vec<TokenId>) -> Vec<TokenId> {
        while symbols.len() >= 2 {
            let mut best_cand: Option<(usize, TokenId, u32)> = None;
            for i in 0..(symbols.len() - 1) {
                let pair = Pair::new(symbols[i], symbols[i + 1]);
                if let Some(&(merged_id, rank)) = self.merge_ranks.get(&pair) {
                    match best_cand {
                        None => best_cand = Some((i, merged_id, rank)),
                        Some((_, _, best_rank)) if rank < best_rank => {
                            best_cand = Some((i, merged_id, rank));
                        }
                        _ => {}
                    }
                }
            }

            match best_cand {
                None => break,
                Some((idx, merged_id, _)) => {
                    symbols[idx] = merged_id;
                    symbols.remove(idx + 1);
                }
            }
        }
        symbols
    }
}
