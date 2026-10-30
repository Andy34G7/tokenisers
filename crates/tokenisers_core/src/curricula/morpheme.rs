use ahash::{AHashMap, AHashSet};
use std::fmt::Debug;

use crate::index::Pair;
use crate::scoring::{PairContext, PairScorer, ScoreDeps};

/// Lexicon mapping word byte slices to known morphological boundary byte offsets.
#[derive(Debug, Clone, Default)]
pub struct MorphemeLexicon {
    boundaries: AHashMap<Vec<u8>, AHashSet<usize>>,
}

impl MorphemeLexicon {
    pub fn new() -> Self {
        Self {
            boundaries: AHashMap::default(),
        }
    }

    /// Inserts a word and its morpheme segmentation into the lexicon.
    /// Example: word = b"unhappiness", morphemes = &[b"un", b"happi", b"ness"] -> boundaries = {2, 7}
    pub fn insert_segmentation(&mut self, word: &[u8], morphemes: &[&[u8]]) {
        let mut offsets = AHashSet::default();
        let mut curr_offset = 0;
        let num_morphemes = morphemes.len().saturating_sub(1);
        for morpheme in morphemes.iter().take(num_morphemes) {
            curr_offset += morpheme.len();
            offsets.insert(curr_offset);
        }
        self.boundaries.insert(word.to_vec(), offsets);
    }

    #[inline(always)]
    pub fn get_boundaries(&self, word: &[u8]) -> Option<&AHashSet<usize>> {
        self.boundaries.get(word)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.boundaries.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.boundaries.is_empty()
    }
}

/// Statistics tracking intra-morpheme vs cross-boundary occurrences for candidate pairs.
#[derive(Debug, Clone, Copy, Default)]
pub struct PairMorphologyStats {
    pub n_intra: u64,
    pub n_cross: u64,
}

/// Morpheme-Guided Prior Scorer wrapper.
/// Modulates candidate pair scores by exp(lambda * M(u, v)), where M(u, v) in [-1.0, 1.0]
/// measures linguistic concordance with known morpheme segmentations.
#[derive(Debug)]
pub struct MorphemeGuidedScorer<S: PairScorer> {
    pub inner_scorer: S,
    pub lambda: f64,
    pub gamma: f64,
    pub stats: AHashMap<Pair, PairMorphologyStats>,
}

impl<S: PairScorer> MorphemeGuidedScorer<S> {
    pub fn new(inner_scorer: S, lambda: f64, gamma: f64) -> Self {
        Self {
            inner_scorer,
            lambda: lambda.max(0.0),
            gamma: gamma.max(0.0),
            stats: AHashMap::default(),
        }
    }

    /// Records an occurrence of pair (u, v) at a junction within `word`.
    pub fn record_occurrence(
        &mut self,
        pair: Pair,
        word: &[u8],
        junction_offset: usize,
        lexicon: &MorphemeLexicon,
    ) {
        if let Some(boundaries) = lexicon.get_boundaries(word) {
            let entry = self.stats.entry(pair).or_default();
            if boundaries.contains(&junction_offset) {
                entry.n_cross += 1;
            } else {
                entry.n_intra += 1;
            }
        }
    }

    /// Directly sets morphology statistics for a pair.
    pub fn set_stats(&mut self, pair: Pair, n_intra: u64, n_cross: u64) {
        self.stats.insert(pair, PairMorphologyStats { n_intra, n_cross });
    }

    /// Computes the normalized concordance M(u, v) in [-1.0, 1.0].
    /// +1.0 = strictly intra-morphemic; -1.0 = strictly cross-boundary; 0.0 = neutral/unannotated.
    pub fn concordance(&self, pair: Pair) -> f64 {
        if let Some(stat) = self.stats.get(&pair) {
            let total = stat.n_intra + stat.n_cross;
            if total == 0 {
                return 0.0;
            }
            let numerator = (stat.n_intra as f64) - self.gamma * (stat.n_cross as f64);
            let denominator = total as f64;
            (numerator / denominator).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

impl<S: PairScorer> PairScorer for MorphemeGuidedScorer<S> {
    fn score(&self, pair: Pair, ctx: &PairContext) -> f64 {
        let base_score = self.inner_scorer.score(pair, ctx);
        if !base_score.is_finite() || self.lambda == 0.0 {
            return base_score;
        }

        let m = self.concordance(pair);
        let modulation = (self.lambda * m).exp();
        if base_score >= 0.0 {
            base_score * modulation
        } else {
            base_score / modulation
        }
    }

    fn name(&self) -> &'static str {
        "morpheme_guided"
    }

    /// The modulation is a fixed positive factor per pair, so dependencies are the inner scorer's.
    fn deps(&self) -> ScoreDeps {
        self.inner_scorer.deps()
    }
}
