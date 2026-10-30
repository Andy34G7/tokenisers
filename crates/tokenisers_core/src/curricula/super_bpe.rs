use crate::engine::bottom_up::{BottomUpModel, TrainState};
use crate::error::{Result, TokeniserError};
use crate::scoring::PairScorer;
use crate::vocab::BASE_VOCAB_SIZE;

/// Configuration for the Two-Stage SuperBPE curriculum.
#[derive(Debug, Clone)]
pub struct SuperBpeConfig {
    pub v_switch: usize,
    pub v_final: usize,
}

impl SuperBpeConfig {
    pub fn new(v_switch: usize, v_final: usize) -> Result<Self> {
        if v_switch < BASE_VOCAB_SIZE {
            return Err(TokeniserError::VocabSizeTooSmall(v_switch));
        }
        if v_switch > v_final {
            return Err(TokeniserError::InvalidConfiguration(format!(
                "v_switch ({v_switch}) cannot exceed v_final ({v_final})"
            )));
        }
        Ok(Self { v_switch, v_final })
    }
}

/// Boundary junction between two segments in the arena.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryJunction {
    pub left_word_idx: u32,
    pub right_word_idx: u32,
    pub freq: u64,
}

/// Two-Stage SuperBPE training engine.
pub struct SuperBpeTrainer<'a> {
    pub config: SuperBpeConfig,
    pub scorer: &'a dyn PairScorer,
    pub min_frequency: u64,
}

impl<'a> SuperBpeTrainer<'a> {
    pub fn new(config: SuperBpeConfig, scorer: &'a dyn PairScorer) -> Self {
        Self {
            config,
            scorer,
            min_frequency: 1,
        }
    }

    /// Splits text into alternating maximal runs of whitespace and non-whitespace characters.
    ///
    /// Whitespace is Unicode `White_Space` (`char::is_whitespace`), so U+3000, U+00A0, U+2028 etc.
    /// separate words just like ASCII space. This is THE segmentation used for training and must
    /// be reused verbatim by any encoder that pre-splits text (see [`Self::segment_bytes`]).
    pub fn segment_text(text: &str) -> Vec<Vec<u8>> {
        Self::segment_bytes(text.as_bytes())
            .into_iter()
            .map(<[u8]>::to_vec)
            .collect()
    }

    /// Byte-level form of [`Self::segment_text`]: identical on valid UTF-8; bytes that are not
    /// valid UTF-8 count as non-whitespace. Concatenating the returned slices gives back `bytes`.
    pub fn segment_bytes(bytes: &[u8]) -> Vec<&[u8]> {
        // (is_whitespace, byte length) per char; each invalid UTF-8 run is one non-ws unit.
        let units = bytes.utf8_chunks().flat_map(|chunk| {
            let invalid = chunk.invalid().len();
            chunk
                .valid()
                .chars()
                .map(|c| (c.is_whitespace(), c.len_utf8()))
                .chain((invalid > 0).then_some((false, invalid)))
        });
        let mut segments = Vec::new();
        let (mut start, mut pos) = (0, 0);
        let mut prev_ws: Option<bool> = None;
        for (ws, len) in units {
            if prev_ws.is_some_and(|p| p != ws) {
                segments.push(&bytes[start..pos]);
                start = pos;
            }
            prev_ws = Some(ws);
            pos += len;
        }
        if pos > start {
            segments.push(&bytes[start..pos]);
        }
        segments
    }

    /// Trains Two-Stage SuperBPE on a collection of document lines.
    ///
    /// Every segment occurrence is its own word. Stage 1 merges inside words until the vocabulary
    /// reaches `v_switch` (or no pair is left); stage 2 then links neighbouring segments of each
    /// line and keeps merging, now across whitespace, up to `v_final`. Both stages run the
    /// `BottomUpTrainer` engine, so every step merges the scorer's exact argmax. Morpheme
    /// annotations (`PairScorer::word_boundaries`) are per word, so SuperBPE ignores them.
    pub fn train(&self, lines: &[&str]) -> Result<BottomUpModel> {
        if lines.is_empty() {
            return Err(TokeniserError::EmptyCorpus);
        }
        let mut state = TrainState::new(self.scorer, self.min_frequency, false);
        let mut junctions = Vec::new();
        for line in lines {
            let mut prev = None;
            for seg in Self::segment_bytes(line.as_bytes()) {
                let word = state.add_word(seg, 1);
                if let Some(left) = prev {
                    junctions.push(BoundaryJunction {
                        left_word_idx: left,
                        right_word_idx: word,
                        freq: 1,
                    });
                }
                prev = Some(word);
            }
        }
        state.queue_all();

        let mut merges = Vec::new();
        let mut stage2 = false;
        while state.vocab.len() < self.config.v_final {
            if !stage2 && state.vocab.len() >= self.config.v_switch {
                log::info!("SuperBPE: lifting whitespace boundaries at v_switch = {}", self.config.v_switch);
                stage2 = true;
                state.link_words(&junctions);
            }
            match state.pop_best() {
                Some(pair) => {
                    let new_token = state.apply_merge(pair)?;
                    merges.push((pair.0, pair.1, new_token));
                }
                None if !stage2 => {
                    log::info!(
                        "SuperBPE: early transition to stage 2 (intra-word pairs exhausted at vocab len {})",
                        state.vocab.len()
                    );
                    stage2 = true;
                    state.link_words(&junctions);
                }
                None => break,
            }
        }

        let mut model = BottomUpModel::new(state.vocab, merges);
        // Stage-2 merges may span whitespace, so such a model must encode whole text.
        model.pretokenize = !stage2;
        Ok(model)
    }
}
