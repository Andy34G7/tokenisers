use fastrand::Rng;

use crate::engine::unigram::{CompactTrie, logsumexp2};
use crate::vocab::TokenId;

/// An edge in the subword lattice.
#[derive(Debug, Clone, Copy)]
pub struct LatticeEdge {
    pub start_pos: usize,
    pub end_pos: usize,
    pub token_id: TokenId,
    pub log_prob: f64,
}

/// Stochastic Unigram Lattice Sampler supporting exact FFBS and k-best path sampling.
pub struct UnigramSampler<'a> {
    pub trie: &'a CompactTrie,
}

impl<'a> UnigramSampler<'a> {
    pub fn new(trie: &'a CompactTrie) -> Self {
        Self { trie }
    }

    /// Constructs the incoming lattice edges for each position in the byte sequence.
    pub fn build_incoming_lattice(&self, bytes: &[u8]) -> Vec<Vec<LatticeEdge>> {
        let n = bytes.len();
        let mut incoming = vec![Vec::new(); n + 1];

        for start in 0..n {
            let matches = self.trie.common_prefix_search(&bytes[start..]);
            for (len, token_id, log_prob) in matches {
                let end = start + len;
                incoming[end].push(LatticeEdge {
                    start_pos: start,
                    end_pos: end,
                    token_id,
                    log_prob,
                });
            }
        }

        // Guarantee fallback connectivity at any unreached position
        for i in 1..=n {
            if incoming[i].is_empty() {
                // Base byte fallback
                let byte_val = bytes[i - 1];
                incoming[i].push(LatticeEdge {
                    start_pos: i - 1,
                    end_pos: i,
                    token_id: byte_val as TokenId,
                    log_prob: -15.0,
                });
            }
        }

        incoming
    }

    /// Forward-Filtering Backward-Sampling (FFBS) from the exact lattice posterior.
    pub fn sample_ffbs(&self, bytes: &[u8], temperature: f64, seed: Option<u64>) -> Vec<TokenId> {
        let n = bytes.len();
        if n == 0 {
            return Vec::new();
        }

        let incoming = self.build_incoming_lattice(bytes);
        let inv_temp = 1.0 / temperature.max(1e-4);

        // Forward Pass: compute log_alpha(i)
        let mut log_alpha = vec![f64::NEG_INFINITY; n + 1];
        log_alpha[0] = 0.0;

        for i in 1..=n {
            for edge in &incoming[i] {
                let prev_alpha = log_alpha[edge.start_pos];
                if prev_alpha > f64::NEG_INFINITY {
                    let weight = prev_alpha + inv_temp * edge.log_prob;
                    log_alpha[i] = logsumexp2(log_alpha[i], weight);
                }
            }
        }

        // Backward Sampling Pass
        let mut rng = Rng::new();
        if let Some(s) = seed {
            rng.seed(s);
        }

        let mut curr_pos = n;
        let mut tokens = Vec::new();

        while curr_pos > 0 {
            let edges = &incoming[curr_pos];
            let mut probs = Vec::with_capacity(edges.len());

            for edge in edges {
                let prev_alpha = log_alpha[edge.start_pos];
                let logit = prev_alpha + inv_temp * edge.log_prob - log_alpha[curr_pos];
                probs.push(logit.exp().max(0.0));
            }

            let sum_prob: f64 = probs.iter().sum();
            let mut choice_val = if sum_prob > 0.0 {
                rng.f64() * sum_prob
            } else {
                0.0
            };

            let mut selected_idx = 0;
            for (idx, &p) in probs.iter().enumerate() {
                choice_val -= p;
                if choice_val <= 0.0 {
                    selected_idx = idx;
                    break;
                }
            }

            let chosen_edge = &edges[selected_idx];
            tokens.push(chosen_edge.token_id);
            curr_pos = chosen_edge.start_pos;
        }

        tokens.reverse();
        tokens
    }

    /// k-Best shortest path sampling via multi-backpointer Viterbi.
    pub fn sample_kbest(&self, bytes: &[u8], k: usize, temperature: f64, seed: Option<u64>) -> Vec<TokenId> {
        let n = bytes.len();
        if n == 0 {
            return Vec::new();
        }

        let k = k.max(1);
        let incoming = self.build_incoming_lattice(bytes);

        // PathEntry: (score, prev_pos, token_id, prev_rank)
        #[derive(Clone, Debug)]
        struct PathEntry {
            score: f64,
            prev_pos: usize,
            token_id: TokenId,
            prev_rank: usize,
        }

        let mut dp: Vec<Vec<PathEntry>> = vec![Vec::new(); n + 1];
        dp[0].push(PathEntry {
            score: 0.0,
            prev_pos: 0,
            token_id: 0,
            prev_rank: 0,
        });

        for i in 1..=n {
            let mut cands = Vec::new();
            for edge in &incoming[i] {
                for (prev_rank, prev_entry) in dp[edge.start_pos].iter().enumerate() {
                    let total_score = prev_entry.score + edge.log_prob;
                    cands.push(PathEntry {
                        score: total_score,
                        prev_pos: edge.start_pos,
                        token_id: edge.token_id,
                        prev_rank,
                    });
                }
            }

            cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            cands.truncate(k);
            dp[i] = cands;
        }

        let terminal_paths = &dp[n];
        if terminal_paths.is_empty() {
            return bytes.iter().map(|&b| b as TokenId).collect();
        }

        if k == 1 {
            // Strictly deterministic Viterbi 1-best
            let mut tokens = Vec::new();
            let mut curr_pos = n;
            let mut curr_rank = 0;
            while curr_pos > 0 {
                let entry = &dp[curr_pos][curr_rank];
                tokens.push(entry.token_id);
                curr_pos = entry.prev_pos;
                curr_rank = entry.prev_rank;
            }
            tokens.reverse();
            return tokens;
        }

        let inv_temp = 1.0 / temperature.max(1e-4);
        let max_score = terminal_paths[0].score;
        let unnorm_probs: Vec<f64> = terminal_paths
            .iter()
            .map(|p| (inv_temp * (p.score - max_score)).exp())
            .collect();
        let sum_p: f64 = unnorm_probs.iter().sum();

        let mut rng = Rng::new();
        if let Some(s) = seed {
            rng.seed(s);
        }

        let mut r = rng.f64() * sum_p;
        let mut chosen_rank = 0;
        for (idx, &p) in unnorm_probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                chosen_rank = idx;
                break;
            }
        }

        let mut tokens = Vec::new();
        let mut curr_pos = n;
        let mut curr_rank = chosen_rank;

        while curr_pos > 0 {
            let entry = &dp[curr_pos][curr_rank];
            tokens.push(entry.token_id);
            curr_pos = entry.prev_pos;
            curr_rank = entry.prev_rank;
        }

        tokens.reverse();
        tokens
    }
}
