use ahash::AHashMap;
use std::fmt::Debug;

use crate::index::Pair;
use crate::scoring::{PairContext, PairScorer, ScoreDeps};

/// Lexicon mapping training words (exact bytes of one pre-tokenized segment) to their morpheme
/// boundaries: ascending byte offsets strictly inside the word. The caller supplies it (gold
/// training annotations or unsupervised boundaries); nothing here reads evaluation data.
#[derive(Debug, Clone, Default)]
pub struct MorphemeLexicon {
    boundaries: AHashMap<Vec<u8>, Vec<usize>>,
}

impl MorphemeLexicon {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a word and its surface segmentation.
    /// Example: word = b"unhappiness", morphemes = &[b"un", b"happi", b"ness"] -> boundaries [2, 7].
    ///
    /// Returns false and inserts nothing when the morphemes do not concatenate to `word` (e.g. a
    /// canonical segmentation such as "happiness" -> "happy" + "ness"): their lengths would not be
    /// byte offsets into the word.
    pub fn insert_segmentation(&mut self, word: &[u8], morphemes: &[&[u8]]) -> bool {
        if morphemes.concat() != word {
            return false;
        }
        let mut offsets = Vec::with_capacity(morphemes.len());
        let mut offset = 0;
        for morpheme in morphemes {
            offset += morpheme.len();
            if offset > 0 && offset < word.len() && offsets.last() != Some(&offset) {
                offsets.push(offset);
            }
        }
        self.boundaries.insert(word.to_vec(), offsets);
        true
    }

    #[inline(always)]
    pub fn get_boundaries(&self, word: &[u8]) -> Option<&[usize]> {
        self.boundaries.get(word).map(Vec::as_slice)
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

/// Frequency-weighted counts of a pair's current occurrences inside annotated words. An
/// occurrence is `cross` when the token the merge would create spans a morpheme boundary of its
/// word, else `intra`. The training engine maintains these per pair (see
/// [`PairScorer::word_boundaries`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PairMorphologyStats {
    pub n_intra: u64,
    pub n_cross: u64,
}

/// Morpheme-Guided Prior Scorer wrapper.
///
/// Scales the inner score by exp(lambda * M(u, v)), where M in [-1, 1] is the concordance of the
/// pair's current occurrences in lexicon words: (n_intra - gamma * n_cross) / (n_intra + n_cross),
/// clamped. Pairs that never occur in a lexicon word have M = 0 and keep the inner score, so an
/// empty lexicon trains exactly like the inner scorer. A pair whose reweighted count
/// freq(uv) * exp(lambda * M) falls below one occurrence is not merged at all, so a strong lambda
/// never learns a token across a known boundary, even when nothing else is left to merge.
#[derive(Debug)]
pub struct MorphemeGuidedScorer<S: PairScorer> {
    pub inner_scorer: S,
    pub lexicon: MorphemeLexicon,
    pub lambda: f64,
    pub gamma: f64,
}

impl<S: PairScorer> MorphemeGuidedScorer<S> {
    pub fn new(inner_scorer: S, lexicon: MorphemeLexicon, lambda: f64, gamma: f64) -> Self {
        Self {
            inner_scorer,
            lexicon,
            lambda: lambda.max(0.0),
            gamma: gamma.max(0.0),
        }
    }

    /// Normalized concordance M in [-1.0, 1.0]: +1.0 = strictly intra-morphemic, -1.0 = strictly
    /// cross-boundary, 0.0 = no annotated occurrence.
    pub fn concordance(&self, stats: PairMorphologyStats) -> f64 {
        let total = stats.n_intra + stats.n_cross;
        if total == 0 {
            return 0.0;
        }
        let numerator = stats.n_intra as f64 - self.gamma * stats.n_cross as f64;
        (numerator / total as f64).clamp(-1.0, 1.0)
    }
}

impl<S: PairScorer> PairScorer for MorphemeGuidedScorer<S> {
    /// Without occurrence statistics the pair counts as unannotated (M = 0).
    fn score(&self, pair: Pair, ctx: &PairContext) -> f64 {
        self.inner_scorer.score(pair, ctx)
    }

    fn name(&self) -> &'static str {
        "morpheme_guided"
    }

    /// The modulation reads only the pair's own occurrence statistics, which change exactly when
    /// its count does, so the dependencies are the inner scorer's.
    fn deps(&self) -> ScoreDeps {
        self.inner_scorer.deps()
    }

    fn word_boundaries(&self, word: &[u8]) -> Option<&[usize]> {
        self.lexicon.get_boundaries(word)
    }

    fn score_morph(&self, pair: Pair, ctx: &PairContext, morph: PairMorphologyStats) -> f64 {
        let base_score = self.inner_scorer.score(pair, ctx);
        let m = self.concordance(morph);
        if !base_score.is_finite() || self.lambda == 0.0 || m == 0.0 {
            return base_score;
        }
        // Kept finite and positive: a huge lambda must not turn scores into inf (ineligible) or NaN.
        let modulation = (self.lambda * m).exp().clamp(f64::MIN_POSITIVE, f64::MAX);
        if (ctx.freq_uv as f64) * modulation < 1.0 {
            return f64::NEG_INFINITY;
        }
        let scaled = if base_score >= 0.0 {
            base_score * modulation
        } else {
            base_score / modulation
        };
        scaled.clamp(-f64::MAX, f64::MAX)
    }
}
