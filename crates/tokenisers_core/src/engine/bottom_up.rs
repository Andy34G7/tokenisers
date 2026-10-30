use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use ahash::{AHashMap, AHashSet};
use fastrand::Rng;
use rayon::prelude::*;

use crate::arena::{SymbolArena, NULL_NODE};
use crate::curricula::morpheme::PairMorphologyStats;
use crate::curricula::super_bpe::SuperBpeTrainer;
use crate::error::{Result, TokeniserError};
use crate::index::{InvertedIndex, OccurrenceRef, Pair};
use crate::scoring::{PairContext, PairScorer, ScoreDeps, ScoredPair};
use crate::vocab::{TokenId, Vocabulary, BASE_VOCAB_SIZE};

const NONE: u32 = u32::MAX;

/// Trained Bottom-Up Model containing vocabulary and ordered merge operations.
///
/// `merges[r] = (u, v, w)` is the merge applied at rank `r`. Different merges may produce the same
/// token id (byte strings are deduplicated) and the same pair may be merged at more than one rank;
/// encoding reproduces both exactly.
#[derive(Clone, Debug)]
pub struct BottomUpModel {
    pub vocab: Vocabulary,
    pub merges: Vec<(TokenId, TokenId, TokenId)>,
    /// First rank at which each pair is merged.
    pub merge_ranks: AHashMap<Pair, u32>,
    /// `next_same_rank[r]`: next rank that merges the same pair as rank `r` (`u32::MAX` if none).
    next_same_rank: Vec<u32>,
    /// True when training only ever merged inside `SuperBpeTrainer::segment_text` segments (every
    /// training word was one segment), so no learned token spans a segment boundary. Encoding then
    /// pre-splits input with `SuperBpeTrainer::segment_bytes` and caches repeated segments, and the
    /// HF export emits the matching `Split` pre-tokenizer. False (the `new` default, SuperBPE
    /// stage 2, models rebuilt from a bare merge list): each special-token-free span of the input
    /// is one merge unit and the export is whole-text `ByteLevel`.
    pub pretokenize: bool,
}

/// Reusable per-call buffers for the heap encoder.
#[derive(Default)]
struct Scratch {
    /// (token, prev, next); token == NONE marks a removed symbol.
    syms: Vec<(TokenId, u32, u32)>,
    heap: BinaryHeap<Reverse<(u32, u32)>>,
    skipped: Vec<Reverse<(u32, u32)>>,
}

impl BottomUpModel {
    pub fn new(vocab: Vocabulary, merges: Vec<(TokenId, TokenId, TokenId)>) -> Self {
        let mut merge_ranks: AHashMap<Pair, u32> = AHashMap::with_capacity(merges.len());
        let mut last_rank: AHashMap<Pair, u32> = AHashMap::new();
        let mut next_same_rank = vec![NONE; merges.len()];
        for (rank, &(u, v, _)) in merges.iter().enumerate() {
            let pair = Pair::new(u, v);
            let rank = rank as u32;
            merge_ranks.entry(pair).or_insert(rank);
            if let Some(prev) = last_rank.insert(pair, rank) {
                next_same_rank[prev as usize] = rank;
            }
        }
        Self {
            vocab,
            merges,
            merge_ranks,
            next_same_rank,
            pretokenize: false,
        }
    }

    /// Smallest rank >= `min_rank` that merges `pair`.
    #[inline]
    fn rank_from(&self, pair: Pair, min_rank: u32) -> Option<u32> {
        let mut rank = *self.merge_ranks.get(&pair)?;
        while rank < min_rank {
            rank = self.next_same_rank[rank as usize];
            if rank == NONE {
                return None;
            }
        }
        Some(rank)
    }

    /// Deterministic BPE encoding: special tokens are matched first (leftmost-longest), then merges
    /// are applied in training order. O(n log n) in the input length.
    pub fn encode(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.encode_impl(bytes, None)
    }

    /// BPE-dropout encoding (Provilkov et al. 2020, HF `tokenizers` semantics): within each merge
    /// unit (segment for `pretokenize` models), every popped merge is skipped with probability `p` and
    /// re-queued after the next successful merge. `p <= 0` equals `encode`, `p >= 1` yields raw
    /// bytes (special tokens are still matched). A seed makes the output reproducible.
    pub fn encode_with_dropout(&self, bytes: &[u8], p: f32, seed: Option<u64>) -> Vec<TokenId> {
        if p.is_nan() || p <= 0.0 {
            return self.encode(bytes);
        }
        let mut rng = seed.map_or_else(Rng::new, Rng::with_seed);
        self.encode_impl(bytes, Some((p.min(1.0), &mut rng)))
    }

    fn encode_impl(&self, bytes: &[u8], mut dropout: Option<(f32, &mut Rng)>) -> Vec<TokenId> {
        let mut out = Vec::with_capacity(bytes.len() / 2 + 1);
        let mut scratch = Scratch::default();
        let mut cache: AHashMap<&[u8], (usize, usize)> = AHashMap::new();
        let specials = self.vocab.special_tokens();

        let mut start = 0;
        let mut i = 0;
        while i < bytes.len() && !specials.is_empty() {
            let hit = specials
                .iter()
                .filter(|s| !s.bytes.is_empty() && bytes[i..].starts_with(&s.bytes))
                .max_by_key(|s| s.bytes.len());
            match hit {
                Some(special) => {
                    self.encode_span(&bytes[start..i], &mut dropout, &mut scratch, &mut cache, &mut out);
                    out.push(special.id);
                    i += special.bytes.len();
                    start = i;
                }
                None => i += 1,
            }
        }
        self.encode_span(&bytes[start..], &mut dropout, &mut scratch, &mut cache, &mut out);
        out
    }

    /// Encodes a span free of special tokens.
    fn encode_span<'b>(
        &self,
        span: &'b [u8],
        dropout: &mut Option<(f32, &mut Rng)>,
        scratch: &mut Scratch,
        cache: &mut AHashMap<&'b [u8], (usize, usize)>,
        out: &mut Vec<TokenId>,
    ) {
        if span.is_empty() {
            return;
        }
        if !self.pretokenize {
            return self.merge_unit(span, dropout, scratch, out);
        }
        // Byte-level segmentation: equals `segment_text` on valid UTF-8 and also splits invalid
        // input, so a byte-level merge can never straddle a multi-byte whitespace character.
        for piece in SuperBpeTrainer::segment_bytes(span) {
            if dropout.is_some() {
                self.merge_unit(piece, dropout, scratch, out);
            } else if let Some(&(s, e)) = cache.get(piece) {
                out.extend_from_within(s..e);
            } else {
                let s = out.len();
                self.merge_unit(piece, dropout, scratch, out);
                cache.insert(piece, (s, out.len()));
            }
        }
    }

    /// Applies merges to one unit with a min-heap over (rank, position) and a linked list of
    /// symbols, so equal ranks resolve leftmost-first exactly like training.
    fn merge_unit(
        &self,
        bytes: &[u8],
        dropout: &mut Option<(f32, &mut Rng)>,
        scratch: &mut Scratch,
        out: &mut Vec<TokenId>,
    ) {
        let n = bytes.len();
        if n < 2 || self.merges.is_empty() {
            out.extend(bytes.iter().map(|&b| b as TokenId));
            return;
        }
        let Scratch { syms, heap, skipped } = scratch;
        syms.clear();
        heap.clear();
        skipped.clear();
        for (i, &b) in bytes.iter().enumerate() {
            let prev = if i == 0 { NONE } else { i as u32 - 1 };
            let next = if i + 1 == n { NONE } else { i as u32 + 1 };
            syms.push((b as TokenId, prev, next));
        }
        for i in 0..n - 1 {
            let pair = Pair::new(bytes[i] as TokenId, bytes[i + 1] as TokenId);
            if let Some(rank) = self.rank_from(pair, 0) {
                heap.push(Reverse((rank, i as u32)));
            }
        }

        while let Some(top) = heap.pop() {
            if let Some((p, rng)) = dropout.as_mut() {
                if rng.f32() < *p {
                    skipped.push(top);
                    continue;
                }
            }
            heap.extend(skipped.drain(..));

            let Reverse((rank, pos)) = top;
            let (tok, prev, next) = syms[pos as usize];
            if tok == NONE || next == NONE {
                continue;
            }
            let (u, v, w) = self.merges[rank as usize];
            if tok != u || syms[next as usize].0 != v {
                continue; // stale entry
            }
            let after = syms[next as usize].2;
            syms[pos as usize] = (w, prev, after);
            syms[next as usize].0 = NONE;
            if after != NONE {
                syms[after as usize].1 = pos;
            }
            if prev != NONE {
                if let Some(r) = self.rank_from(Pair::new(syms[prev as usize].0, w), rank + 1) {
                    heap.push(Reverse((r, prev)));
                }
            }
            if after != NONE {
                if let Some(r) = self.rank_from(Pair::new(w, syms[after as usize].0), rank + 1) {
                    heap.push(Reverse((r, pos)));
                }
            }
        }

        // Position 0 is never removed: a merge keeps its left symbol.
        let mut cur = 0u32;
        while cur != NONE {
            out.push(syms[cur as usize].0);
            cur = syms[cur as usize].2;
        }
    }

    /// Parallel batch encoding across multiple texts.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<TokenId>> {
        texts
            .par_iter()
            .map(|text| self.encode(text.as_bytes()))
            .collect()
    }

    /// Decodes token IDs back to raw bytes; errors on ids outside the vocabulary.
    pub fn decode(&self, ids: &[TokenId]) -> Result<Vec<u8>> {
        self.vocab.decode(ids)
    }

    /// Decodes token IDs back to a UTF-8 String (byte lossless).
    pub fn decode_lossless(&self, ids: &[TokenId]) -> Result<String> {
        self.vocab.decode_lossless(ids)
    }
}

/// Pre-tokenizes text lines in parallel into word frequency counts using Rayon.
pub fn parallel_count_words(lines: &[&str]) -> AHashMap<Vec<u8>, u64> {
    lines
        .par_chunks(2048)
        .map(|chunk| {
            let mut local_counts: AHashMap<Vec<u8>, u64> = AHashMap::default();
            for line in chunk {
                // Word splitting on whitespace preserving content
                for word in line.split_whitespace() {
                    *local_counts.entry(word.as_bytes().to_vec()).or_insert(0) += 1;
                }
            }
            local_counts
        })
        .reduce(AHashMap::default, |mut map1, map2| {
            for (word, count) in map2 {
                *map1.entry(word).or_insert(0) += count;
            }
            map1
        })
}

/// Unified Bottom-Up training engine supporting arbitrary PairScorers, O(1) splicing,
/// and version-tombstoned priority queues.
///
/// Every step merges the exact argmax of `scorer.score` over all current pairs (ties: smaller
/// pair ids first), i.e. the merge a full recount of pair counts, token counts and N followed by
/// re-scoring every pair would pick. After a merge (u, v) -> w the engine re-scores every pair whose
/// count changed and, for scorers that read marginals, every pair containing u, v or w (token ->
/// pairs index). Scorers that also read N (`ScoreDeps::MonotoneTotals`) never gain score as N
/// shrinks, so their stale heap entries are upper bounds: a popped entry is accepted only once it
/// was scored at the current step and every stale entry within float tolerance of it was refreshed.
pub struct BottomUpTrainer<'a> {
    pub scorer: &'a dyn PairScorer,
    pub target_vocab_size: usize,
    pub min_frequency: u64,
}

impl<'a> BottomUpTrainer<'a> {
    pub fn new(scorer: &'a dyn PairScorer, target_vocab_size: usize) -> Self {
        Self {
            scorer,
            target_vocab_size,
            min_frequency: 1,
        }
    }

    pub fn with_min_frequency(mut self, min_frequency: u64) -> Self {
        self.min_frequency = min_frequency;
        self
    }

    /// Trains a Bottom-Up tokenizer from unique words with their corpus frequencies.
    pub fn train(&self, word_counts: &[(Vec<u8>, u64)]) -> Result<BottomUpModel> {
        if self.target_vocab_size < BASE_VOCAB_SIZE {
            return Err(TokeniserError::VocabSizeTooSmall(self.target_vocab_size));
        }
        if word_counts.is_empty() {
            return Err(TokeniserError::EmptyCorpus);
        }

        let mut state = TrainState::new(self.scorer, self.min_frequency.max(1), word_counts);
        let mut merges = Vec::new();
        while state.vocab.len() < self.target_vocab_size {
            let Some(pair) = state.pop_best() else { break };
            let new_token = state.apply_merge(pair)?;
            merges.push((pair.0, pair.1, new_token));
        }

        let pretokenize = word_counts.iter().all(|(word, _)| {
            std::str::from_utf8(word).is_ok_and(|text| SuperBpeTrainer::segment_text(text).len() <= 1)
        });
        let mut model = BottomUpModel::new(state.vocab, merges);
        model.pretokenize = pretokenize;
        Ok(model)
    }
}

#[derive(Clone, Copy, Default)]
struct PairStat {
    count: u64,
    version: u32,
    /// Last step at which this pair was queued for re-scoring (dedupe marker).
    touched: u32,
}

/// Heap entry: `ScoredPair` ordering, then the fresher step first.
#[derive(Clone, Copy)]
struct Entry {
    scored: ScoredPair,
    step: u32,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.scored.cmp(&other.scored).then(self.step.cmp(&other.step))
    }
}

struct TrainState<'s> {
    scorer: &'s dyn PairScorer,
    deps: ScoreDeps,
    min_frequency: u64,
    vocab: Vocabulary,
    arena: SymbolArena,
    index: InvertedIndex,
    stats: AHashMap<Pair, PairStat>,
    live_pairs: usize,
    token_counts: Vec<u64>,
    /// token -> pairs with count > 0 containing it (kept only when scores read marginals).
    pairs_of: Vec<AHashSet<Pair>>,
    track_pairs_of: bool,
    total_tokens: u64,
    heap: BinaryHeap<Entry>,
    /// Number of merges applied so far.
    step: u32,
    /// Pairs to re-score after the current merge.
    changed: Vec<Pair>,
    /// Per arena word: its morpheme boundaries (`PairScorer::word_boundaries`). Empty when no
    /// training word is annotated, which turns morphology tracking off.
    word_bounds: Vec<Option<&'s [usize]>>,
    /// Per pair: its current occurrences inside annotated words, split intra / cross-boundary.
    morph: AHashMap<Pair, PairMorphologyStats>,
}

impl<'s> TrainState<'s> {
    fn new(scorer: &'s dyn PairScorer, min_frequency: u64, word_counts: &[(Vec<u8>, u64)]) -> Self {
        let deps = scorer.deps();
        let track_pairs_of = matches!(deps, ScoreDeps::Marginals | ScoreDeps::MonotoneTotals);
        // Same filter as the arena loop below, so index = word id.
        let mut word_bounds: Vec<Option<&'s [usize]>> = word_counts
            .iter()
            .filter(|(word, freq)| !word.is_empty() && *freq > 0)
            .map(|(word, _)| scorer.word_boundaries(word))
            .collect();
        if word_bounds.iter().all(Option::is_none) {
            word_bounds.clear();
        }
        let mut state = Self {
            scorer,
            deps,
            min_frequency,
            vocab: Vocabulary::new_byte_level(),
            arena: SymbolArena::with_capacity(word_counts.len() * 8, word_counts.len()),
            index: InvertedIndex::with_capacity(word_counts.len() * 4),
            stats: AHashMap::default(),
            live_pairs: 0,
            token_counts: vec![0; BASE_VOCAB_SIZE],
            pairs_of: vec![AHashSet::default(); if track_pairs_of { BASE_VOCAB_SIZE } else { 0 }],
            track_pairs_of,
            total_tokens: 0,
            heap: BinaryHeap::new(),
            step: 0,
            changed: Vec::new(),
            word_bounds,
            morph: AHashMap::default(),
        };

        for (word_bytes, freq) in word_counts {
            if word_bytes.is_empty() || *freq == 0 {
                continue;
            }
            let tokens: Vec<TokenId> = word_bytes.iter().map(|&b| b as TokenId).collect();
            for &tok in &tokens {
                state.token_counts[tok as usize] += *freq;
                state.total_tokens += *freq;
            }
            let word_id = state.arena.add_word(&tokens, *freq);
            let head = state.arena.get_word(word_id).unwrap().head;
            for (i, window) in tokens.windows(2).enumerate() {
                let pair = Pair::new(window[0], window[1]);
                state.count_occurrence(pair, word_id, head + i as u32, *freq as i64);
                state.index.add_occurrence(pair, OccurrenceRef::new(word_id, head + i as u32, 0));
            }
        }

        let pairs: Vec<Pair> = state.stats.keys().copied().collect();
        for pair in pairs {
            state.push(pair);
        }
        state
    }

    /// Adds `delta` to a pair's count, keeping `pairs_of` / `live_pairs` in sync and queueing the
    /// pair for re-scoring after the current merge.
    fn add_count(&mut self, pair: Pair, delta: i64) {
        let stat = self.stats.entry(pair).or_default();
        let old = stat.count;
        stat.count = if delta >= 0 {
            old + delta as u64
        } else {
            old.saturating_sub(delta.unsigned_abs())
        };
        let new = stat.count;
        if self.step > 0 && stat.touched != self.step {
            stat.touched = self.step;
            self.changed.push(pair);
        }
        if (old == 0) != (new == 0) {
            if new == 0 {
                self.live_pairs -= 1;
            } else {
                self.live_pairs += 1;
            }
            if self.track_pairs_of {
                for tok in [pair.0, pair.1] {
                    if new == 0 {
                        self.pairs_of[tok as usize].remove(&pair);
                    } else {
                        self.pairs_of[tok as usize].insert(pair);
                    }
                }
            }
        }
    }

    /// `add_count` for the occurrence of `pair` whose left symbol is node `left` of arena word
    /// `word`, also updating the pair's morphology statistics if the word is annotated.
    fn count_occurrence(&mut self, pair: Pair, word: u32, left: u32, delta: i64) {
        self.add_count(pair, delta);
        let Some(&Some(bounds)) = self.word_bounds.get(word as usize) else {
            return;
        };
        // Words start with one node per byte and a merge keeps its left node, so a symbol's node
        // index minus the word head is its byte offset. The merged token spans [start, end).
        let start = (left - self.arena.get_word(word).unwrap().head) as usize;
        let token_len = |t: TokenId| self.vocab.get_bytes(t).map_or(1, <[u8]>::len);
        let end = start + token_len(pair.0) + token_len(pair.1);
        let cross = bounds.iter().any(|&b| start < b && b < end);
        let stat = self.morph.entry(pair).or_default();
        let n = if cross { &mut stat.n_cross } else { &mut stat.n_intra };
        *n = n.saturating_add_signed(delta);
    }

    fn score(&self, pair: Pair, count: u64) -> f64 {
        let ctx = PairContext {
            freq_uv: count,
            freq_u: self.token_counts[pair.0 as usize],
            freq_v: self.token_counts[pair.1 as usize],
            total_tokens: self.total_tokens,
            active_vocab_size: self.vocab.len(),
            len_u: self.vocab.get_bytes(pair.0).map_or(1, |b| b.len()),
            len_v: self.vocab.get_bytes(pair.1).map_or(1, |b| b.len()),
        };
        if self.word_bounds.is_empty() {
            return self.scorer.score(pair, &ctx);
        }
        let morph = self.morph.get(&pair).copied().unwrap_or_default();
        self.scorer.score_morph(pair, &ctx, morph)
    }

    /// Pushes an entry scored now for `pair` at its current version, if eligible.
    fn push(&mut self, pair: Pair) {
        let stat = self.stats[&pair];
        if stat.count < self.min_frequency {
            return;
        }
        let score = self.score(pair, stat.count);
        if score.is_finite() {
            self.heap.push(Entry {
                scored: ScoredPair { score, pair, version: stat.version },
                step: self.step,
            });
        }
    }

    fn is_live(&self, entry: &Entry) -> bool {
        let stat = self.stats[&entry.scored.pair];
        stat.version == entry.scored.version && stat.count >= self.min_frequency
    }

    /// Pops the exact best pair (see `BottomUpTrainer` for why lazy refresh is exact).
    fn pop_best(&mut self) -> Option<Pair> {
        let lazy = self.deps == ScoreDeps::MonotoneTotals;
        let mut banded = false;
        while let Some(entry) = self.heap.pop() {
            if !self.is_live(&entry) {
                continue;
            }
            if !lazy || banded {
                return Some(entry.scored.pair);
            }
            if entry.step != self.step {
                self.push(entry.scored.pair);
                continue;
            }
            // `entry` is exact; stale entries are upper bounds up to float rounding. Refresh every
            // stale entry within a small band below it, then pop again: the new top is exact and
            // beats every remaining stale bound.
            banded = true;
            let score = entry.scored.score;
            let floor = score - 1e-9 * (score.abs() + 1.0);
            let mut fresh = vec![entry];
            while self.heap.peek().is_some_and(|top| top.scored.score >= floor) {
                let top = self.heap.pop().unwrap();
                if !self.is_live(&top) {
                    continue;
                }
                if top.step == self.step {
                    fresh.push(top);
                } else {
                    self.push(top.scored.pair);
                }
            }
            self.heap.extend(fresh);
        }
        None
    }

    /// Merges every occurrence of `pair` (left to right within each word) and re-queues the
    /// candidates whose score may have changed.
    fn apply_merge(&mut self, pair: Pair) -> Result<TokenId> {
        let (u, v) = (pair.0, pair.1);
        let mut merged_bytes = self.vocab.get_bytes(u).unwrap().to_vec();
        merged_bytes.extend_from_slice(self.vocab.get_bytes(v).unwrap());
        let w = self.vocab.add_token(merged_bytes);
        if w as usize >= self.token_counts.len() {
            self.token_counts.resize(w as usize + 1, 0);
            if self.track_pairs_of {
                self.pairs_of.resize_with(w as usize + 1, AHashSet::default);
            }
        }

        self.step += 1;
        self.changed.clear();
        let remaining = self.stats[&pair].count;
        self.add_count(pair, -(remaining as i64));
        self.morph.remove(&pair);

        let mut occurrences = self.index.take(&pair).unwrap_or_default();
        // Arena indices grow left to right inside a word, so overlapping occurrences (e.g. "aaa")
        // are merged leftmost-first, exactly like encoding does.
        occurrences.sort_unstable_by_key(|occ| occ.node_idx);

        let mut merged = 0u64;
        for occ in occurrences {
            if !occ.is_valid(pair, &self.arena) {
                continue;
            }
            let left_idx = occ.node_idx;
            let left = *self.arena.get_node(left_idx).unwrap();
            let right_idx = left.next;
            let next_idx = self.arena.get_node(right_idx).unwrap().next;
            let prev_idx = left.prev;
            let word_id = occ.word_id;
            let freq = self.arena.get_word(word_id).unwrap().freq;

            if prev_idx != NULL_NODE {
                let prev_tok = self.arena.get_token(prev_idx);
                self.count_occurrence(Pair::new(prev_tok, u), word_id, prev_idx, -(freq as i64));
            }
            if next_idx != NULL_NODE {
                let next_tok = self.arena.get_token(next_idx);
                self.count_occurrence(Pair::new(v, next_tok), word_id, right_idx, -(freq as i64));
            }

            self.arena.splice_pair(left_idx, right_idx, w, word_id)?;

            if prev_idx != NULL_NODE {
                let prev_node = *self.arena.get_node(prev_idx).unwrap();
                let new_pair = Pair::new(prev_node.token, w);
                self.count_occurrence(new_pair, word_id, prev_idx, freq as i64);
                self.index
                    .add_occurrence(new_pair, OccurrenceRef::new(word_id, prev_idx, prev_node.version));
            }
            if next_idx != NULL_NODE {
                let new_pair = Pair::new(w, self.arena.get_token(next_idx));
                self.count_occurrence(new_pair, word_id, left_idx, freq as i64);
                let version = self.arena.get_version(left_idx);
                self.index
                    .add_occurrence(new_pair, OccurrenceRef::new(word_id, left_idx, version));
            }
            merged += freq;
        }

        self.token_counts[u as usize] -= merged;
        self.token_counts[v as usize] -= merged; // u == v: 2 * merged in total
        self.token_counts[w as usize] += merged;
        self.total_tokens -= merged;

        if self.track_pairs_of {
            // The marginals of u, v and w changed: every pair containing them may move.
            let step = self.step;
            for tok in [u, v, w] {
                for &p in &self.pairs_of[tok as usize] {
                    let stat = self.stats.get_mut(&p).unwrap();
                    if stat.touched != step {
                        stat.touched = step;
                        self.changed.push(p);
                    }
                }
            }
        }

        let changed = std::mem::take(&mut self.changed);
        for &p in &changed {
            let stat = self.stats.get_mut(&p).unwrap();
            stat.version = stat.version.wrapping_add(1);
        }
        if self.deps == ScoreDeps::Arbitrary {
            // ponytail: full re-score per merge; only for scorers that declare no dependencies.
            self.heap.clear();
            let live: Vec<Pair> = self.stats.iter().filter(|(_, s)| s.count > 0).map(|(&p, _)| p).collect();
            for p in live {
                self.push(p);
            }
        } else {
            for &p in &changed {
                self.push(p);
            }
        }
        self.changed = changed;

        // Drop dead entries once they dominate the heap.
        if self.heap.len() > 4 * self.live_pairs + (1 << 16) {
            let stats = &self.stats;
            let min_frequency = self.min_frequency;
            self.heap.retain(|e| {
                let s = stats[&e.scored.pair];
                s.version == e.scored.version && s.count >= min_frequency
            });
        }

        Ok(w)
    }
}
