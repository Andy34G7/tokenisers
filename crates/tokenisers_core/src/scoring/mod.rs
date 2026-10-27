pub mod frequency;
pub mod pmi;
pub mod mdl;
pub mod length_penalized;

use std::cmp::Ordering;
use std::fmt::Debug;
use crate::index::Pair;

pub use frequency::BpeScorer;
pub use pmi::WordPieceScorer;
pub use mdl::MdlScorer;
pub use length_penalized::LengthPenalizedScorer;

/// Snapshot of active corpus and pair statistics supplied to `PairScorer::score`.
#[derive(Clone, Copy, Debug)]
pub struct PairContext {
    /// Active co-occurrence count of pair (u, v) in corpus.
    pub freq_uv: u64,
    /// Marginal frequency of token u in corpus.
    pub freq_u: u64,
    /// Marginal frequency of token v in corpus.
    pub freq_v: u64,
    /// Total active token count in corpus.
    pub total_tokens: u64,
    /// Number of active unique tokens in vocabulary.
    pub active_vocab_size: usize,
    /// Byte length of token u.
    pub len_u: usize,
    /// Byte length of token v.
    pub len_v: usize,
}

impl PairContext {
    #[inline(always)]
    pub fn len_uv(&self) -> usize {
        self.len_u + self.len_v
    }
}

/// Pluggable interface for evaluating merge priority of adjacent symbol pairs.
pub trait PairScorer: Send + Sync + Debug {
    /// Evaluates merge score for `pair` under the provided `ctx`.
    /// Higher values indicate higher merge priority.
    /// Returns `f64::NEG_INFINITY` if the pair is ineligible for merging.
    fn score(&self, pair: Pair, ctx: &PairContext) -> f64;

    /// Canonical identifier for serialization and diagnostics.
    fn name(&self) -> &'static str;
}

/// Candidate pair stored in max-heap priority queue with version tombstoning and deterministic tie-breaking.
#[derive(Clone, Copy, Debug)]
pub struct ScoredPair {
    pub score: f64,
    pub pair: Pair,
    pub version: u32,
}

impl PartialEq for ScoredPair {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.pair == other.pair && self.version == other.version
    }
}

impl Eq for ScoredPair {}

impl PartialOrd for ScoredPair {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredPair {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: highest score pops first. If score is NaN, treat as lowest.
        match self.score.partial_cmp(&other.score) {
            Some(Ordering::Equal) | None => {
                // Deterministic secondary tie-breaker:
                // Prefer smaller token IDs (other.pair.0.cmp(&self.pair.0))
                other.pair.0.cmp(&self.pair.0)
                    .then_with(|| other.pair.1.cmp(&self.pair.1))
                    .then_with(|| self.version.cmp(&other.version))
            }
            Some(ord) => ord,
        }
    }
}
