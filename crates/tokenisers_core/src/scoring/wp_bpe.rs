use crate::index::Pair;
use super::{PairContext, PairScorer, ScoreDeps};

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

    fn deps(&self) -> ScoreDeps {
        ScoreDeps::Marginals
    }
}

/// Frequency-Thresholded PMI scorer: freq(uv) / (freq(u) * freq(v)), only for freq(uv) >= min_frequency
/// (at least 2). Same function as `WpBpeBaseScorer` except for the cutoff (and as `WordPieceScorer`
/// with alpha = 0).
#[derive(Clone, Copy, Debug)]
pub struct WpBpeFtScorer {
    pub min_frequency: u64,
    pub epsilon: f64,
}

impl Default for WpBpeFtScorer {
    fn default() -> Self {
        Self {
            min_frequency: 2,
            epsilon: 1e-12,
        }
    }
}

impl WpBpeFtScorer {
    pub fn new(min_frequency: u64) -> Self {
        Self {
            min_frequency: min_frequency.max(2),
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
        let denom = ((ctx.freq_u as f64) * (ctx.freq_v as f64)) + self.epsilon;

        num / denom
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "wp_bpe_ft"
    }

    fn deps(&self) -> ScoreDeps {
        ScoreDeps::Marginals
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

    fn deps(&self) -> ScoreDeps {
        ScoreDeps::MonotoneTotals
    }
}

/// Symmetric Conditional Probability (SCP) / Geometric-Mean Calibrated PMI scorer with Kneser-Ney style discounting:
/// Score = ((freq_uv as f64) - self.discount).max(0.0) / (((freq_u as f64) * (freq_v as f64)).sqrt() + 1e-12)
#[derive(Clone, Copy, Debug)]
pub struct WpBpeScpScorer {
    pub min_freq: u64,
    pub discount: f64,
    pub epsilon: f64,
}

impl Default for WpBpeScpScorer {
    fn default() -> Self {
        Self {
            min_freq: 2,
            discount: 0.75,
            epsilon: 1e-12,
        }
    }
}

impl WpBpeScpScorer {
    pub fn new(min_freq: u64, discount: f64) -> Self {
        Self {
            min_freq: min_freq.max(1),
            discount: discount.max(0.0),
            epsilon: 1e-12,
        }
    }
}

impl PairScorer for WpBpeScpScorer {
    #[inline(always)]
    fn score(&self, _pair: Pair, ctx: &PairContext) -> f64 {
        if ctx.freq_uv < self.min_freq {
            return f64::NEG_INFINITY;
        }

        let denom = ((ctx.freq_u as f64) * (ctx.freq_v as f64)).sqrt() + self.epsilon;
        let num = ((ctx.freq_uv as f64) - self.discount).max(0.0);
        num / denom
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "wp_bpe_scp"
    }

    fn deps(&self) -> ScoreDeps {
        ScoreDeps::Marginals
    }
}

/// Normalized Pointwise Mutual Information (NPMI, Bouma 2009) scorer:
/// score = freq(uv)^alpha * PMI(u, v) / (-ln P(uv)), with P estimated as counts / N.
/// NPMI is in [-1, 1]; negatively associated pairs keep a (negative) finite score, so they stay
/// eligible and rank below every positively associated pair.
#[derive(Clone, Copy, Debug)]
pub struct WpBpeNpmiScorer {
    pub min_freq: u64,
    pub alpha: f64,
    pub epsilon: f64,
}

impl Default for WpBpeNpmiScorer {
    fn default() -> Self {
        Self {
            min_freq: 2,
            alpha: 0.5,
            epsilon: 1e-12,
        }
    }
}

impl WpBpeNpmiScorer {
    pub fn new(min_freq: u64, alpha: f64) -> Self {
        Self {
            min_freq: min_freq.max(1),
            alpha: alpha.max(0.0),
            epsilon: 1e-12,
        }
    }
}

impl PairScorer for WpBpeNpmiScorer {
    #[inline(always)]
    fn score(&self, _pair: Pair, ctx: &PairContext) -> f64 {
        if ctx.freq_uv < self.min_freq {
            return f64::NEG_INFINITY;
        }

        let n = (ctx.total_tokens as f64).max(1.0);
        let denom_uv = ((ctx.freq_u as f64) * (ctx.freq_v as f64)) + self.epsilon;
        let pmi_ratio = ((ctx.freq_uv as f64) * n / denom_uv).max(self.epsilon);
        let pmi = pmi_ratio.ln();

        let p_uv = ((ctx.freq_uv as f64) / n).clamp(self.epsilon, 1.0);
        let self_info = -p_uv.ln();

        let npmi = if self_info > 1e-12 {
            pmi / self_info
        } else {
            0.0
        };

        (ctx.freq_uv as f64).powf(self.alpha) * npmi
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "wp_bpe_npmi"
    }

    fn deps(&self) -> ScoreDeps {
        ScoreDeps::MonotoneTotals
    }
}

/// Positive Length-Regularized Pointwise Information Gain (PPIG) scorer:
/// score(u, v) = freq(uv) * max(0, PMI(u, v)) / (len_u + len_v)^length_penalty
#[derive(Clone, Copy, Debug)]
pub struct WpBpePpigScorer {
    pub min_freq: u64,
    pub length_penalty: f64,
    pub epsilon: f64,
}

impl Default for WpBpePpigScorer {
    fn default() -> Self {
        Self {
            min_freq: 2,
            length_penalty: 0.35,
            epsilon: 1e-12,
        }
    }
}

impl WpBpePpigScorer {
    pub fn new(min_freq: u64, length_penalty: f64) -> Self {
        Self {
            min_freq: min_freq.max(1),
            length_penalty: length_penalty.max(0.0),
            epsilon: 1e-12,
        }
    }
}

impl PairScorer for WpBpePpigScorer {
    #[inline(always)]
    fn score(&self, _pair: Pair, ctx: &PairContext) -> f64 {
        if ctx.freq_uv < self.min_freq {
            return f64::NEG_INFINITY;
        }

        let n = (ctx.total_tokens as f64).max(1.0);
        let denom = ((ctx.freq_u as f64) * (ctx.freq_v as f64)) + self.epsilon;
        let pmi_ratio = ((ctx.freq_uv as f64) * n / denom).max(self.epsilon);
        let pmi = pmi_ratio.log2().max(0.0);

        let merged_len = (ctx.len_uv().max(1) as f64).powf(self.length_penalty);
        ((ctx.freq_uv as f64) * pmi) / merged_len
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "wp_bpe_ppig"
    }

    fn deps(&self) -> ScoreDeps {
        ScoreDeps::MonotoneTotals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_context(freq_uv: u64, freq_u: u64, freq_v: u64, total_tokens: u64, len_u: usize, len_v: usize) -> PairContext {
        PairContext {
            freq_uv,
            freq_u,
            freq_v,
            total_tokens,
            active_vocab_size: 256,
            len_u,
            len_v,
        }
    }

    #[test]
    fn test_wp_bpe_scp_scorer() {
        let scorer = WpBpeScpScorer::new(2, 0.75);
        let pair = Pair::new(10, 20);

        // Below min_freq
        let ctx_low = mock_context(1, 10, 10, 100, 1, 1);
        assert_eq!(scorer.score(pair, &ctx_low), f64::NEG_INFINITY);

        // Valid pair
        let ctx_valid = mock_context(10, 20, 30, 200, 1, 1);
        let score = scorer.score(pair, &ctx_valid);
        assert!(score > 0.0);

        // Higher frequency uv yields higher score
        let ctx_higher = mock_context(15, 20, 30, 200, 1, 1);
        assert!(scorer.score(pair, &ctx_higher) > score);
        assert_eq!(scorer.name(), "wp_bpe_scp");
    }

    #[test]
    fn test_wp_bpe_npmi_scorer() {
        let scorer = WpBpeNpmiScorer::new(2, 0.5);
        let pair = Pair::new(10, 20);

        // Below min_freq
        let ctx_low = mock_context(1, 10, 10, 100, 1, 1);
        assert_eq!(scorer.score(pair, &ctx_low), f64::NEG_INFINITY);

        // Negatively associated pair (pmi < 0): eligible, negative NPMI
        let ctx_discorr = mock_context(2, 100, 100, 200, 1, 1);
        let score_discorr = scorer.score(pair, &ctx_discorr);
        assert!(score_discorr.is_finite() && score_discorr < 0.0);

        // High collocation pair
        let ctx_colloc = mock_context(50, 60, 60, 1000, 1, 1);
        let score = scorer.score(pair, &ctx_colloc);
        assert!(score > 0.0);
        assert_eq!(scorer.name(), "wp_bpe_npmi");
    }

    #[test]
    fn test_wp_bpe_ppig_scorer() {
        let scorer = WpBpePpigScorer::new(2, 0.35);
        let pair = Pair::new(10, 20);

        // Below min_freq
        let ctx_low = mock_context(1, 10, 10, 100, 1, 1);
        assert_eq!(scorer.score(pair, &ctx_low), f64::NEG_INFINITY);

        // Negatively associated pair: max(0, PMI) = 0, still eligible
        let ctx_discorr = mock_context(2, 100, 100, 200, 1, 1);
        assert_eq!(scorer.score(pair, &ctx_discorr), 0.0);

        // High collocation pair
        let ctx_colloc = mock_context(50, 60, 60, 1000, 1, 1);
        let score = scorer.score(pair, &ctx_colloc);
        assert!(score > 0.0);

        // Longer merged pair receives length penalty
        let ctx_long = mock_context(50, 60, 60, 1000, 4, 4);
        assert!(scorer.score(pair, &ctx_long) < score);
        assert_eq!(scorer.name(), "wp_bpe_ppig");
    }
}
