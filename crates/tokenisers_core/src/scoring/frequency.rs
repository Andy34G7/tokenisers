use crate::index::Pair;
use super::{PairContext, PairScorer};

/// Standard Byte-Level BPE frequency scorer: score(u, v) = freq(uv).
#[derive(Clone, Copy, Debug, Default)]
pub struct BpeScorer;

pub type FrequencyScorer = BpeScorer;

impl BpeScorer {
    pub fn new() -> Self {
        Self
    }
}

impl PairScorer for BpeScorer {
    #[inline(always)]
    fn score(&self, _pair: Pair, ctx: &PairContext) -> f64 {
        if ctx.freq_uv == 0 {
            f64::NEG_INFINITY
        } else {
            ctx.freq_uv as f64
        }
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "bpe"
    }
}
