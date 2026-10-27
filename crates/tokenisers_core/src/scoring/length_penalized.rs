use crate::index::Pair;
use super::{PairContext, PairScorer};

/// Length-Penalized / Length-Weighted Frequency Scorer:
/// score(u, v) = freq(uv) * (|u| + |v|)^beta.
#[derive(Clone, Copy, Debug)]
pub struct LengthPenalizedScorer {
    /// Exponent applied to merged byte length: (|u| + |v|)^beta.
    /// beta > 0 rewards longer subwords; beta < 0 penalizes longer subwords; beta = 0 matches BPE.
    pub beta: f64,
    /// Minimum frequency required to consider merging.
    pub min_frequency: u64,
}

impl Default for LengthPenalizedScorer {
    fn default() -> Self {
        Self {
            beta: 0.5,
            min_frequency: 1,
        }
    }
}

impl LengthPenalizedScorer {
    pub fn new(beta: f64, min_frequency: u64) -> Self {
        Self {
            beta,
            min_frequency,
        }
    }
}

impl PairScorer for LengthPenalizedScorer {
    #[inline(always)]
    fn score(&self, _pair: Pair, ctx: &PairContext) -> f64 {
        if ctx.freq_uv < self.min_frequency {
            return f64::NEG_INFINITY;
        }

        let len_uv = ctx.len_uv() as f64;
        let factor = if self.beta == 0.0 {
            1.0
        } else {
            len_uv.powf(self.beta)
        };

        (ctx.freq_uv as f64) * factor
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "length_penalized"
    }
}
