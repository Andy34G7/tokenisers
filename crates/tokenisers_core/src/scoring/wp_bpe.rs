use crate::index::Pair;
use super::{PairContext, PairScorer};

/// Uncalibrated Pointwise Mutual Information (PMI) scorer: freq(uv) / (freq(u) * freq(v)).
/// min_frequency is 1 (uncalibrated, no cutoff on rare pairs).
#[derive(Clone, Copy, Debug)]
pub struct WpBpeBaseScorer {
    pub epsilon: f64,
}

impl Default for WpBpeBaseScorer {
    fn default() -> Self {
        Self { epsilon: 1e-12 }
    }
}

impl WpBpeBaseScorer {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PairScorer for WpBpeBaseScorer {
    #[inline(always)]
    fn score(&self, _pair: Pair, ctx: &PairContext) -> f64 {
        if ctx.freq_uv == 0 {
            return f64::NEG_INFINITY;
        }

        let num = ctx.freq_uv as f64;
        let denom_u = ctx.freq_u as f64;
        let denom_v = ctx.freq_v as f64;
        let denom = (denom_u * denom_v) + self.epsilon;

        num / denom
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "wp_bpe_base"
    }
}

/// Frequency-Thresholded PMI scorer:
/// Evaluates PMI only if freq(uv) >= min_frequency (default threshold >= 2 or user-specified).
#[derive(Clone, Copy, Debug)]
pub struct WpBpeFtScorer {
    pub min_frequency: u64,
    pub alpha: f64,
    pub epsilon: f64,
}

impl Default for WpBpeFtScorer {
    fn default() -> Self {
        Self {
            min_frequency: 2,
            alpha: 1.0,
            epsilon: 1e-12,
        }
    }
}

impl WpBpeFtScorer {
    pub fn new(min_frequency: u64, alpha: f64) -> Self {
        Self {
            min_frequency: min_frequency.max(2),
            alpha: alpha.max(0.0),
            epsilon: 1e-12,
        }
    }
}

impl PairScorer for WpBpeFtScorer {
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
        "wp_bpe_ft"
    }
}

/// Pointwise Information Gain (PIG) scorer:
/// score(u, v) = freq(uv) * log2((freq(uv) * total_tokens) / (freq(u) * freq(v)))
/// Measures total corpus mutual information contribution (co-occurrence frequency * PMI).
#[derive(Clone, Copy, Debug)]
pub struct WpBpePigScorer {
    pub min_frequency: u64,
    pub epsilon: f64,
}

impl Default for WpBpePigScorer {
    fn default() -> Self {
        Self {
            min_frequency: 1,
            epsilon: 1e-12,
        }
    }
}

impl WpBpePigScorer {
    pub fn new(min_frequency: u64) -> Self {
        Self {
            min_frequency: min_frequency.max(1),
            epsilon: 1e-12,
        }
    }
}

impl PairScorer for WpBpePigScorer {
    #[inline(always)]
    fn score(&self, _pair: Pair, ctx: &PairContext) -> f64 {
        if ctx.freq_uv < self.min_frequency {
            return f64::NEG_INFINITY;
        }

        let num = (ctx.freq_uv as f64) * (ctx.total_tokens.max(1) as f64);
        let denom = ((ctx.freq_u as f64) * (ctx.freq_v as f64)) + self.epsilon;
        let ratio = (num / denom).max(1e-12);
        let pmi = ratio.log2();

        // Information gain = freq(uv) * PMI
        (ctx.freq_uv as f64) * pmi
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "wp_bpe_pig"
    }
}
