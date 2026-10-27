use crate::index::Pair;
use super::{PairContext, PairScorer};

/// Minimum Description Length (MDL / Delta-Bits) information-theoretic two-part code compression scorer.
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
}

impl PairScorer for MdlScorer {
    fn score(&self, _pair: Pair, ctx: &PairContext) -> f64 {
        if ctx.freq_uv < self.min_frequency || ctx.total_tokens == 0 {
            return f64::NEG_INFINITY;
        }

        let c_uv = ctx.freq_uv as f64;
        let c_u = ctx.freq_u as f64;
        let c_v = ctx.freq_v as f64;
        let n = ctx.total_tokens as f64;

        let n_prime = (ctx.total_tokens.saturating_sub(ctx.freq_uv)) as f64;
        let c_u_prime = (ctx.freq_u.saturating_sub(ctx.freq_uv)) as f64;
        let c_v_prime = (ctx.freq_v.saturating_sub(ctx.freq_uv)) as f64;

        let delta_n = Self::x_log2_x(n) - Self::x_log2_x(n_prime);
        let delta_u = Self::x_log2_x(c_u) - Self::x_log2_x(c_u_prime);
        let delta_v = Self::x_log2_x(c_v) - Self::x_log2_x(c_v_prime);
        let gain_uv = Self::x_log2_x(c_uv);

        let delta_corpus_bits = delta_n - delta_u - delta_v + gain_uv;
        let delta_vocab_bits = self.entry_overhead_bits + (ctx.len_uv() as f64 * 8.0);

        delta_corpus_bits - delta_vocab_bits
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "mdl"
    }
}
