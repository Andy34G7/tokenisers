use crate::index::Pair;
use super::{PairContext, PairScorer, ScoreDeps};

/// Laplace-smoothed WordPiece / PMI scorer:
/// score(u, v) = freq(uv) / ((freq(u) + alpha) * (freq(v) + alpha)), alpha = 1 by default.
///
/// The smoothing is what separates it from the Gen-1 variants: with alpha = 0 it is the same
/// function as `WpBpeBaseScorer` (min_frequency 1) and `WpBpeFtScorer` (min_frequency >= 2).
#[derive(Clone, Copy, Debug)]
pub struct WordPieceScorer {
    /// Additive smoothing on the marginals (0 = unsmoothed PMI ratio).
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

    fn deps(&self) -> ScoreDeps {
        ScoreDeps::Marginals
    }
}
