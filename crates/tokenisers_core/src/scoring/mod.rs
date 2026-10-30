pub mod frequency;
pub mod pmi;
pub mod mdl;
pub mod length_penalized;
pub mod wp_bpe;

use std::cmp::Ordering;
use std::fmt::Debug;
use crate::curricula::morpheme::PairMorphologyStats;
use crate::index::Pair;

pub use frequency::BpeScorer;
pub use pmi::WordPieceScorer;
pub use mdl::MdlScorer;
pub use length_penalized::LengthPenalizedScorer;
pub use wp_bpe::{
    WpBpeBaseScorer, WpBpeFtScorer, WpBpeNpmiScorer, WpBpePigScorer, WpBpePpigScorer,
    WpBpeScpScorer,
};

/// Snapshot of active corpus and pair statistics supplied to `PairScorer::score`.
#[derive(Clone, Copy, Debug)]
pub struct PairContext {
    /// Active co-occurrence count of pair (u, v) in corpus: every adjacent (u, v) window, so
    /// overlapping windows of a self-pair all count ("aaa" holds two (a, a) windows).
    pub freq_uv: u64,
    /// Number of merges that merging (u, v) applies: its windows taken leftmost-first without
    /// overlap. Equals `freq_uv` unless u == v, where a run of k u's has k - 1 windows but merges
    /// only k / 2 times. Read it for anything that models the corpus after the merge.
    pub merges_uv: u64,
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

/// Which `PairContext` fields a scorer reads. The training engine uses it to re-score exactly the
/// candidates whose score can change after a merge, so every step merges the true argmax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreDeps {
    /// Only `freq_uv`, `len_u`, `len_v`.
    PairOnly,
    /// Also `freq_u` / `freq_v`.
    Marginals,
    /// Also `total_tokens` / `active_vocab_size`. Contract: with `freq_uv`, `freq_u`, `freq_v` and the
    /// lengths fixed, the score must never increase as `total_tokens` shrinks and `active_vocab_size`
    /// grows. A stale score is then an upper bound and the engine re-scores lazily at pop time.
    MonotoneTotals,
    /// Unknown dependencies: every candidate is re-scored after every merge (exact but slow).
    Arbitrary,
}

/// Pluggable interface for evaluating merge priority of adjacent symbol pairs.
pub trait PairScorer: Send + Sync + Debug {
    /// Evaluates merge score for `pair` under the provided `ctx`.
    /// Higher values indicate higher merge priority.
    /// Returns `f64::NEG_INFINITY` if the pair is ineligible for merging.
    fn score(&self, pair: Pair, ctx: &PairContext) -> f64;

    /// Canonical identifier for serialization and diagnostics.
    fn name(&self) -> &'static str;

    /// Statistics this scorer depends on (see [`ScoreDeps`]). The default is always exact but
    /// re-scores every candidate after every merge; declare a narrower set for speed.
    fn deps(&self) -> ScoreDeps {
        ScoreDeps::Arbitrary
    }

    /// Morpheme boundaries of a training word (ascending byte offsets strictly inside it), or
    /// `None` for an unannotated word (the default). When any training word is annotated, the
    /// engine tracks [`PairMorphologyStats`] for every pair over its current occurrences and
    /// scores with [`PairScorer::score_morph`].
    fn word_boundaries(&self, _word: &[u8]) -> Option<&[usize]> {
        None
    }

    /// `score` given the pair's occurrence statistics in annotated words. They change only when
    /// the pair's count changes, so they need no extra [`ScoreDeps`]. The default ignores them.
    fn score_morph(&self, pair: Pair, ctx: &PairContext, _morph: PairMorphologyStats) -> f64 {
        self.score(pair, ctx)
    }
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
