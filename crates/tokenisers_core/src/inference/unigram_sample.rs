use fastrand::Rng;

use crate::curricula::super_bpe::SuperBpeTrainer;
use crate::engine::unigram::{CompactTrie, logsumexp2};
use crate::vocab::TokenId;

/// Upper bound on the k-best table (`segment length * k` path entries, 32 bytes each, so about
/// 32 MB) that [`UnigramSampler::sample_kbest`] builds for one segment.
const MAX_KBEST_ENTRIES: usize = 1 << 20;

/// An edge in the subword lattice.
#[derive(Debug, Clone, Copy)]
pub struct LatticeEdge {
    pub start_pos: usize,
    pub end_pos: usize,
    pub token_id: TokenId,
    pub log_prob: f64,
}

/// Stochastic Unigram Lattice Sampler supporting exact FFBS and k-best path sampling.
///
/// Temperature semantics (both samplers): a segmentation `s` is weighted by
/// `P(s)^(1/temperature) = prod_{t in s} p(t)^(1/temperature)`. `temperature = 1` samples the
/// model's exact posterior; SentencePiece's smoothing parameter is `alpha = 1/temperature`
/// (alpha = 0.1 <=> temperature = 10); `temperature -> 0` approaches Viterbi. No hidden scaling.
/// Like `UnigramModel::encode`, the lattice never spans a `segment_bytes` boundary.
pub struct UnigramSampler<'a> {
    pub trie: &'a CompactTrie,
}

impl<'a> UnigramSampler<'a> {
    pub fn new(trie: &'a CompactTrie) -> Self {
        Self { trie }
    }

    /// Constructs the incoming lattice edges for each position in the byte sequence. Edges stay
    /// inside whitespace/non-whitespace segments, matching training and `UnigramModel::encode`.
    pub fn build_incoming_lattice(&self, bytes: &[u8]) -> Vec<Vec<LatticeEdge>> {
        let n = bytes.len();
        let mut incoming = vec![Vec::new(); n + 1];

        let mut seg_start = 0;
        for seg in SuperBpeTrainer::segment_bytes(bytes) {
            let seg_end = seg_start + seg.len();
            for start in seg_start..seg_end {
                for (len, token_id, log_prob) in self.trie.common_prefix_search(&bytes[start..seg_end]) {
                    let end = start + len;
                    incoming[end].push(LatticeEdge {
                        start_pos: start,
                        end_pos: end,
                        token_id,
                        log_prob,
                    });
                }
            }
            seg_start = seg_end;
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

    /// Forward-Filtering Backward-Sampling: an exact sample from `P(s)^(1/temperature)`
    /// (normalised over all segmentations). `seed = None` draws a fresh random seed.
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

            // Default to the last reachable edge if rounding leaves choice_val slightly > 0.
            let mut selected_idx = probs.iter().rposition(|&p| p > 0.0).unwrap_or(0);
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

    /// Samples one of the `k` highest-scoring segmentations (multi-backpointer Viterbi) with
    /// probability proportional to `P(s)^(1/temperature)` renormalised over those `k`
    /// (SentencePiece `nbest_size = k`). `k = 1` is deterministic Viterbi.
    ///
    /// The k-best choice is made per `segment_bytes` segment (word or whitespace run), like a
    /// k-best encoder applied to each pre-token: the lattice never spans segments, so they are
    /// independent, and each one draws from its own top `k`. This keeps memory at
    /// O(segment length * k) instead of O(text length * k), so a whole document costs no more
    /// than its longest word. A segment of `len` bytes uses
    /// `min(k, MAX_KBEST_ENTRIES / (len + 1))` (at least 1, i.e. Viterbi) so a single huge
    /// segment or a huge `k` cannot exhaust memory.
    pub fn sample_kbest(&self, bytes: &[u8], k: usize, temperature: f64, seed: Option<u64>) -> Vec<TokenId> {
        let mut rng = Rng::new();
        if let Some(s) = seed {
            rng.seed(s);
        }
        let mut tokens = Vec::with_capacity(bytes.len());
        for seg in SuperBpeTrainer::segment_bytes(bytes) {
            // ponytail: fixed table budget, so k shrinks on segments longer than ~1M/k bytes;
            // lazy k-best (Huang & Chiang 2005) would keep the full k if that ever matters.
            let k_seg = k.clamp(1, (MAX_KBEST_ENTRIES / (seg.len() + 1)).max(1));
            tokens.extend(self.sample_kbest_segment(seg, k_seg, temperature, &mut rng));
        }
        tokens
    }

    /// k-best sampling over one segment (`bytes` must not span a `segment_bytes` boundary).
    fn sample_kbest_segment(&self, bytes: &[u8], k: usize, temperature: f64, rng: &mut Rng) -> Vec<TokenId> {
        let n = bytes.len();
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

        let mut cands = Vec::new();
        for i in 1..=n {
            cands.clear();
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
            // Copy out exactly the top k: `cands` holds up to (incoming edges) * k entries.
            dp[i] = cands[..cands.len().min(k)].to_vec();
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

        let mut r = rng.f64() * sum_p;
        let mut chosen_rank = unnorm_probs.len() - 1;
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
