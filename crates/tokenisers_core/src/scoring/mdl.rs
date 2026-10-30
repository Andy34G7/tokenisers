use crate::index::Pair;
use super::{PairContext, PairScorer, ScoreDeps};

/// Minimum Description Length (MDL / Delta-Bits) two-part code scorer.
///
/// Corpus cost is the unigram code length N*log2(N) - sum_t c_t*log2(c_t). Merging (u, v) with
/// count c gives N' = N - c, c_u' = c_u - c, c_v' = c_v - c (c_u' = c_u - 2c when u == v) and
/// c_uv' = c. Score = (corpus bits before - after) - (entry_overhead_bits + 8 * |uv|).
#[derive(Clone, Copy, Debug)]
pub struct MdlScorer {
    /// Fixed bit overhead for adding a new vocabulary entry (metadata/pointer bits).
    /// Default: 24.0 bits (3 bytes).
    pub entry_overhead_bits: f64,
    /// Minimum frequency required to consider merging.
    pub min_frequency: u64,
}

impl Default for MdlScorer {
    fn default() -> Self {
        Self {
            entry_overhead_bits: 24.0,
            min_frequency: 2,
        }
    }
}

impl MdlScorer {
    pub fn new(entry_overhead_bits: f64, min_frequency: u64) -> Self {
        Self {
            entry_overhead_bits: entry_overhead_bits.max(0.0),
            min_frequency,
        }
    }

    #[inline(always)]
    fn x_log2_x(x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            x * x.log2()
        }
    }

    /// x*log2(x) - (x-d)*log2(x-d), written as d*log2(x) - (x-d)*log2(1 - d/x) to avoid the
    /// cancellation of two large terms (keeps the score accurate and monotone in N).
    #[inline(always)]
    fn x_log2_x_drop(x: f64, d: f64) -> f64 {
        if d <= 0.0 {
            0.0
        } else if d >= x {
            Self::x_log2_x(x)
        } else {
            d * x.log2() - (x - d) * (-d / x).ln_1p() / std::f64::consts::LN_2
        }
    }
}

impl PairScorer for MdlScorer {
    fn score(&self, pair: Pair, ctx: &PairContext) -> f64 {
        if ctx.freq_uv < self.min_frequency || ctx.total_tokens == 0 {
            return f64::NEG_INFINITY;
        }

        let c_uv = ctx.freq_uv as f64;
        let delta_n = Self::x_log2_x_drop(ctx.total_tokens as f64, c_uv);
        let delta_marginals = if pair.0 == pair.1 {
            Self::x_log2_x_drop(ctx.freq_u as f64, 2.0 * c_uv)
        } else {
            Self::x_log2_x_drop(ctx.freq_u as f64, c_uv) + Self::x_log2_x_drop(ctx.freq_v as f64, c_uv)
        };
        let gain_uv = Self::x_log2_x(c_uv);

        let delta_corpus_bits = delta_n - delta_marginals + gain_uv;
        let delta_vocab_bits = self.entry_overhead_bits + (ctx.len_uv() as f64 * 8.0);

        delta_corpus_bits - delta_vocab_bits
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "mdl"
    }

    fn deps(&self) -> ScoreDeps {
        ScoreDeps::MonotoneTotals
    }
}
