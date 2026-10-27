use crate::index::Pair;
use super::{PairContext, PairScorer};

/// WordPiece / Pointwise Mutual Information (PMI) scorer with Laplace smoothing.
#[derive(Clone, Copy, Debug)]
pub struct WordPieceScorer {
    /// Additive Laplace smoothing parameter added to marginal token frequencies.
    pub alpha: f64,
    /// Minimum co-occurrence threshold required to consider merging.
    pub min_frequency: u64,
    /// Numerical guard against division by zero.
    pub epsilon: f64,
}

pub type PmiScorer = WordPieceScorer;

impl Default for WordPieceScorer {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            min_frequency: 2,
            epsilon: 1e-12,
        }
    }
}

impl WordPieceScorer {
    pub fn new(alpha: f64, min_frequency: u64) -> Self {
        Self {
            alpha: alpha.max(0.0),
            min_frequency,
            epsilon: 1e-12,
        }
    }
}

impl PairScorer for WordPieceScorer {
    #[inline(always)]
    fn score(&self, _pair: Pair, ctx: &PairContext) -> f64 {
        if ctx.freq_uv < self.min_frequency {
            return f64::NEG_INFINITY;
        }

        let num = ctx.freq_uv as f64;
        let denom_u = ctx.freq_u as f64 + self.alpha;
        let denom_v = ctx.freq_v as f64 + self.alpha;
        let denom = (denom_u * denom_v) + self.epsilon;

        num / denom
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "wordpiece"
    }
}
