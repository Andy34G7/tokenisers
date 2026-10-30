use std::collections::HashSet;
use ahash::AHashMap;
use rayon::prelude::*;

use crate::error::{Result, TokeniserError};
use crate::vocab::{TokenId, Vocabulary, BASE_VOCAB_SIZE};

/// Numerically stable logsumexp for two log-probabilities: ln(exp(a) + exp(b)).
#[inline(always)]
pub fn logsumexp2(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a < 0.0 {
        return b;
    }
    if b.is_infinite() && b < 0.0 {
        return a;
    }
    let max = a.max(b);
    let diff = (a - b).abs();
    if diff > 50.0 {
        max
    } else {
        max + (-diff).exp().ln_1p()
    }
}

/// A node within the arena-allocated CompactPrefixTrie.
#[derive(Clone, Debug)]
pub struct TrieNode {
    pub token_id: Option<TokenId>,
    pub log_prob: f64,
    pub children: Vec<(u8, u32)>,
}

impl Default for TrieNode {
    fn default() -> Self {
        Self {
            token_id: None,
            log_prob: f64::NEG_INFINITY,
            children: Vec::new(),
        }
    }
}

/// Compact Prefix Trie optimized for rapid subword prefix matching over UTF-8 bytes.
#[derive(Clone, Debug)]
pub struct CompactTrie {
    pub nodes: Vec<TrieNode>,
    pub root_table: [Option<u32>; 256],
    pub max_token_len: usize,
}

impl Default for CompactTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactTrie {
    pub fn new() -> Self {
        let mut trie = Self {
            nodes: Vec::with_capacity(65536),
            root_table: [None; 256],
            max_token_len: 0,
        };
        trie.nodes.push(TrieNode::default()); // Root at index 0
        trie
    }

    pub fn build(vocab: &[(Vec<u8>, TokenId, f64)]) -> Self {
        let mut trie = Self::new();
        for (bytes, id, log_prob) in vocab {
            trie.insert(bytes, *id, *log_prob);
        }
        trie
    }

    pub fn insert(&mut self, bytes: &[u8], token_id: TokenId, log_prob: f64) {
        if bytes.is_empty() {
            return;
        }

        self.max_token_len = self.max_token_len.max(bytes.len());
        let first_byte = bytes[0];

        let mut curr_idx = match self.root_table[first_byte as usize] {
            Some(idx) => idx,
            None => {
                let new_idx = self.nodes.len() as u32;
                self.nodes.push(TrieNode::default());
                self.root_table[first_byte as usize] = Some(new_idx);
                new_idx
            }
        };

        for &b in &bytes[1..] {
            curr_idx = self.get_or_create_child(curr_idx, b);
        }

        let node = &mut self.nodes[curr_idx as usize];
        node.token_id = Some(token_id);
        node.log_prob = log_prob;
    }

    #[inline(always)]
    fn get_or_create_child(&mut self, parent_idx: u32, byte: u8) -> u32 {
        let node = &self.nodes[parent_idx as usize];
        for &(b, child_idx) in &node.children {
            if b == byte {
                return child_idx;
            }
        }

        let new_child_idx = self.nodes.len() as u32;
        self.nodes.push(TrieNode::default());

        let node = &mut self.nodes[parent_idx as usize];
        node.children.push((byte, new_child_idx));
        node.children.sort_by_key(|&(b, _)| b);
        new_child_idx
    }

    /// Finds all matching subwords starting at the beginning of `text`.
    /// Returns `(match_len, token_id, log_prob)`.
    #[inline]
    pub fn common_prefix_search(&self, text: &[u8]) -> Vec<(usize, TokenId, f64)> {
        let mut matches = Vec::with_capacity(8);
        if text.is_empty() {
            return matches;
        }

        let first_byte = text[0];
        let mut curr_idx = match self.root_table[first_byte as usize] {
            Some(idx) => idx,
            None => return matches,
        };

        let node = &self.nodes[curr_idx as usize];
        if let Some(id) = node.token_id {
            matches.push((1, id, node.log_prob));
        }

        for (offset, &b) in text[1..].iter().enumerate() {
            let node = &self.nodes[curr_idx as usize];
            let mut next_idx = None;

            for &(edge_byte, child_idx) in &node.children {
                if edge_byte == b {
                    next_idx = Some(child_idx);
                    break;
                }
                if edge_byte > b {
                    break;
                }
            }

            match next_idx {
                Some(idx) => {
                    curr_idx = idx;
                    let child_node = &self.nodes[curr_idx as usize];
                    if let Some(id) = child_node.token_id {
                        matches.push((offset + 2, id, child_node.log_prob));
                    }
                }
                None => break,
            }
        }

        matches
    }
}

/// Dynamic lattice edge discovered during prefix search.
#[derive(Clone, Copy, Debug)]
pub struct LatticeEdge {
    pub start: usize,
    pub end: usize,
    pub token_id: TokenId,
    pub log_prob: f64,
}

/// Viterbi step for 1-best dynamic programming.
#[derive(Clone, Copy, Debug)]
pub struct ViterbiStep {
    pub log_prob: f64,
    pub prev_pos: usize,
    pub token_id: TokenId,
}

pub struct ViterbiDecoder;

impl ViterbiDecoder {
    /// Optimal 1-best Viterbi segmentation for a byte sequence.
    /// If `skip_token` is Some(id), that token is excluded from candidate transitions.
    pub fn decode(
        text: &[u8],
        trie: &CompactTrie,
        skip_token: Option<TokenId>,
    ) -> Option<(f64, Vec<TokenId>)> {
        let n = text.len();
        if n == 0 {
            return Some((0.0, Vec::new()));
        }

        let mut dp = vec![
            ViterbiStep {
                log_prob: f64::NEG_INFINITY,
                prev_pos: 0,
                token_id: 0,
            };
            n + 1
        ];
        dp[0].log_prob = 0.0;

        for start in 0..n {
            if dp[start].log_prob == f64::NEG_INFINITY {
                continue;
            }

            let matches = trie.common_prefix_search(&text[start..]);
            for (len, id, log_prob) in matches {
                if Some(id) == skip_token {
                    continue;
                }

                let end = start + len;
                let candidate_score = dp[start].log_prob + log_prob;

                if candidate_score > dp[end].log_prob {
                    dp[end] = ViterbiStep {
                        log_prob: candidate_score,
                        prev_pos: start,
                        token_id: id,
                    };
                }
            }
        }

        if dp[n].log_prob == f64::NEG_INFINITY {
            return None;
        }

        // Backtrack
        let mut tokens = Vec::new();
        let mut curr = n;
        while curr > 0 {
            tokens.push(dp[curr].token_id);
            curr = dp[curr].prev_pos;
        }
        tokens.reverse();

        Some((dp[n].log_prob, tokens))
    }
}

/// Forward-Backward EM step over pre-tokenized words.
pub struct UnigramEmEngine;

impl UnigramEmEngine {
    /// Computes log-marginal likelihood and posterior expected counts for a single word.
    pub fn forward_backward_word(
        word: &[u8],
        trie: &CompactTrie,
        edges_buf: &mut Vec<LatticeEdge>,
        alpha: &mut Vec<f64>,
        beta: &mut Vec<f64>,
    ) -> (f64, Vec<(TokenId, f64)>) {
        let n = word.len();
        edges_buf.clear();

        for start in 0..n {
            let matches = trie.common_prefix_search(&word[start..]);
            for (len, id, log_prob) in matches {
                edges_buf.push(LatticeEdge {
                    start,
                    end: start + len,
                    token_id: id,
                    log_prob,
                });
            }
        }

        // Forward pass: compute alpha
        alpha.clear();
        alpha.resize(n + 1, f64::NEG_INFINITY);
        alpha[0] = 0.0;

        for edge in edges_buf.iter() {
            if alpha[edge.start] != f64::NEG_INFINITY {
                let term = alpha[edge.start] + edge.log_prob;
                alpha[edge.end] = logsumexp2(alpha[edge.end], term);
            }
        }

        let total_log_prob = alpha[n];
        if total_log_prob == f64::NEG_INFINITY {
            return (f64::NEG_INFINITY, Vec::new());
        }

        // Backward pass: compute beta
        beta.clear();
        beta.resize(n + 1, f64::NEG_INFINITY);
        beta[n] = 0.0;

        for edge in edges_buf.iter().rev() {
            if beta[edge.end] != f64::NEG_INFINITY {
                let term = edge.log_prob + beta[edge.end];
                beta[edge.start] = logsumexp2(beta[edge.start], term);
            }
        }

        // Expected count accumulation
        let mut expected = Vec::with_capacity(edges_buf.len());
        for edge in edges_buf.iter() {
            if alpha[edge.start] != f64::NEG_INFINITY && beta[edge.end] != f64::NEG_INFINITY {
                let posterior_log = alpha[edge.start] + edge.log_prob + beta[edge.end] - total_log_prob;
                let post = posterior_log.exp();
                if post > 1e-12 {
                    expected.push((edge.token_id, post));
                }
            }
        }

        (total_log_prob, expected)
    }

    /// Parallel EM step across corpus words.
    pub fn em_step(
        corpus: &[(Vec<u8>, u64)],
        trie: &CompactTrie,
        vocab_size: usize,
    ) -> (f64, Vec<f64>) {
        let (total_corpus_ll, expected_counts) = corpus
            .par_iter()
            .fold(
                || (0.0f64, vec![0.0f64; vocab_size]),
                |(mut ll, mut counts), (bytes, count)| {
                    let mut edges = Vec::with_capacity(bytes.len() * 4);
                    let mut alpha = Vec::with_capacity(bytes.len() + 1);
                    let mut beta = Vec::with_capacity(bytes.len() + 1);

                    let (word_ll, word_expected) = Self::forward_backward_word(
                        bytes,
                        trie,
                        &mut edges,
                        &mut alpha,
                        &mut beta,
                    );

                    if word_ll.is_finite() {
                        let w_cnt = *count as f64;
                        ll += word_ll * w_cnt;
                        for (id, p) in word_expected {
                            if (id as usize) < counts.len() {
                                counts[id as usize] += p * w_cnt;
                            }
                        }
                    }
                    (ll, counts)
                },
            )
            .reduce(
                || (0.0f64, vec![0.0f64; vocab_size]),
                |(mut ll_a, mut counts_a), (ll_b, counts_b)| {
                    ll_a += ll_b;
                    for (ca, cb) in counts_a.iter_mut().zip(counts_b.iter()) {
                        *ca += *cb;
                    }
                    (ll_a, counts_a)
                },
            );

        // M-step: Laplace smoothed normalization
        let epsilon = 1e-7;
        let mut total_exp = 0.0;
        for &c in &expected_counts {
            total_exp += c + epsilon;
        }

        let ln_total = total_exp.ln();
        let new_log_probs: Vec<f64> = expected_counts
            .iter()
            .map(|&c| (c + epsilon).ln() - ln_total)
            .collect();

        (total_corpus_ll, new_log_probs)
    }
}

/// Trained Unigram Language Model tokenizer.
#[derive(Clone, Debug)]
pub struct UnigramModel {
    pub vocab: Vocabulary,
    pub log_probs: Vec<f64>,
    pub trie: CompactTrie,
}

impl UnigramModel {
    pub fn new(vocab: Vocabulary, log_probs: Vec<f64>) -> Self {
        let indexed_vocab: Vec<(Vec<u8>, TokenId, f64)> = vocab
            .id_to_bytes_map()
            .iter()
            .enumerate()
            .map(|(i, bytes)| (bytes.clone(), i as TokenId, log_probs[i]))
            .collect();
        let trie = CompactTrie::build(&indexed_vocab);
        Self {
            vocab,
            log_probs,
            trie,
        }
    }

    /// Viterbi 1-best segmentation of a byte slice into token IDs.
    pub fn encode(&self, bytes: &[u8]) -> Vec<TokenId> {
        if bytes.is_empty() {
            return Vec::new();
        }
        if let Some((_, tokens)) = ViterbiDecoder::decode(bytes, &self.trie, None) {
            tokens
        } else {
            // Absolute fallback: raw base bytes
            bytes.iter().map(|&b| b as TokenId).collect()
        }
    }

    /// Parallel batch encoding across multiple texts.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<TokenId>> {
        texts
            .par_iter()
            .map(|text| self.encode(text.as_bytes()))
            .collect()
    }

    /// Decodes token IDs back to raw bytes.
    pub fn decode(&self, ids: &[TokenId]) -> Result<Vec<u8>> {
        self.vocab.decode(ids)
    }

    /// Decodes token IDs losslessly to a UTF-8 String.
    pub fn decode_lossless(&self, ids: &[TokenId]) -> Result<String> {
        self.vocab.decode_lossless(ids)
    }
}

/// Unigram LM Trainer implementing log-space EM and loss-based pruning with base-256 byte protection.
pub struct UnigramTrainer {
    pub target_vocab_size: usize,
    pub pruning_rate: f64,
    pub em_sub_iterations: usize,
}

impl Default for UnigramTrainer {
    fn default() -> Self {
        Self {
            target_vocab_size: 16384,
            pruning_rate: 0.15,
            em_sub_iterations: 3,
        }
    }
}

impl UnigramTrainer {
    pub fn new(target_vocab_size: usize, pruning_rate: f64) -> Self {
        Self {
            target_vocab_size,
            pruning_rate: pruning_rate.clamp(0.01, 0.50),
            em_sub_iterations: 3,
        }
    }

    /// Computes marginal likelihood drop for every token across the corpus.
    pub fn compute_pruning_losses(
        corpus: &[(Vec<u8>, u64)],
        trie: &CompactTrie,
        vocab: &[(Vec<u8>, TokenId, f64)],
    ) -> Vec<f64> {
        let vocab_size = vocab.len();

        corpus
            .par_iter()
            .fold(
                || vec![0.0f64; vocab_size],
                |mut thread_losses, (bytes, count)| {
                    if let Some((best_score, best_path)) = ViterbiDecoder::decode(bytes, trie, None) {
                        let unique_tokens: HashSet<TokenId> = best_path.into_iter().collect();

                        for token_id in unique_tokens {
                            let idx = token_id as usize;
                            if idx >= vocab.len() {
                                continue;
                            }

                            // Strict base 256 byte protection: length 1 tokens are never evaluated
                            if vocab[idx].0.len() == 1 {
                                continue;
                            }

                            let w_cnt = *count as f64;
                            if let Some((alt_score, _)) = ViterbiDecoder::decode(bytes, trie, Some(token_id)) {
                                let loss = (best_score - alt_score).max(0.0);
                                thread_losses[idx] += loss * w_cnt;
                            } else {
                                thread_losses[idx] += 1e6 * w_cnt;
                            }
                        }
                    }
                    thread_losses
                },
            )
            .reduce(
                || vec![0.0f64; vocab_size],
                |mut a, b| {
                    for (va, vb) in a.iter_mut().zip(b.iter()) {
                        *va += *vb;
                    }
                    a
                },
            )
    }

    /// Full training loop from seed vocabulary to target vocabulary size.
    pub fn train(&self, corpus: &[(Vec<u8>, u64)]) -> Result<UnigramModel> {
        if self.target_vocab_size < BASE_VOCAB_SIZE {
            return Err(TokeniserError::VocabSizeTooSmall(self.target_vocab_size));
        }
        if corpus.is_empty() {
            return Err(TokeniserError::EmptyCorpus);
        }

        // 1. Build initial seed vocabulary:
        // Always include all 256 base bytes + top frequent character n-grams from corpus
        let mut initial_vocab_map: AHashMap<Vec<u8>, f64> = AHashMap::default();
        for b in 0..=255u8 {
            initial_vocab_map.insert(vec![b], -12.0);
        }

        // Gather substrings of length 2..=16 from corpus
        let seed_target = (self.target_vocab_size * 3).max(BASE_VOCAB_SIZE + 500);
        let mut substring_counts: AHashMap<Vec<u8>, u64> = AHashMap::default();

        for (word_bytes, freq) in corpus {
            let len = word_bytes.len();
            for start in 0..len {
                for end in (start + 2)..=(start + 16).min(len) {
                    let sub = &word_bytes[start..end];
                    *substring_counts.entry(sub.to_vec()).or_insert(0) += *freq;
                }
            }
        }

        let mut sorted_subs: Vec<(Vec<u8>, u64)> = substring_counts.into_iter().collect();
        sorted_subs.sort_by(|a, b| b.1.cmp(&a.1));

        for (sub, _) in sorted_subs.into_iter().take(seed_target - BASE_VOCAB_SIZE) {
            initial_vocab_map.insert(sub, -12.0);
        }

        // Convert to indexed list
        let vocab_list: Vec<(Vec<u8>, f64)> = initial_vocab_map.into_iter().collect();

        // Ensure base bytes are strictly at indices 0..256
        let mut final_seed: Vec<(Vec<u8>, f64)> = Vec::with_capacity(vocab_list.len());
        for b in 0..=255u8 {
            final_seed.push((vec![b], -12.0));
        }
        for (bytes, lp) in vocab_list {
            if bytes.len() > 1 {
                final_seed.push((bytes, lp));
            }
        }

        let mut current_vocab = final_seed;

        // 2. Iterative EM & Pruning loop
        loop {
            let indexed_vocab: Vec<(Vec<u8>, TokenId, f64)> = current_vocab
                .iter()
                .enumerate()
                .map(|(i, (bytes, lp))| (bytes.clone(), i as TokenId, *lp))
                .collect();

            let mut trie = CompactTrie::build(&indexed_vocab);

            // Run EM sub-iterations
            for _ in 0..self.em_sub_iterations {
                let (_, new_log_probs) = UnigramEmEngine::em_step(corpus, &trie, current_vocab.len());
                for (i, &nlp) in new_log_probs.iter().enumerate() {
                    current_vocab[i].1 = nlp;
                }
                let reindexed: Vec<(Vec<u8>, TokenId, f64)> = current_vocab
                    .iter()
                    .enumerate()
                    .map(|(i, (bytes, lp))| (bytes.clone(), i as TokenId, *lp))
                    .collect();
                trie = CompactTrie::build(&reindexed);
            }

            if current_vocab.len() <= self.target_vocab_size {
                break;
            }

            // Pruning step
            let indexed_vocab: Vec<(Vec<u8>, TokenId, f64)> = current_vocab
                .iter()
                .enumerate()
                .map(|(i, (bytes, lp))| (bytes.clone(), i as TokenId, *lp))
                .collect();
            let losses = Self::compute_pruning_losses(corpus, &trie, &indexed_vocab);

            let mut candidates: Vec<(usize, f64)> = losses
                .into_iter()
                .enumerate()
                .map(|(idx, loss)| {
                    if current_vocab[idx].0.len() == 1 {
                        (idx, f64::INFINITY) // Base byte protected
                    } else {
                        (idx, loss)
                    }
                })
                .collect();

            candidates.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| current_vocab[a.0].0.cmp(&current_vocab[b.0].0))
            });

            let current_size = current_vocab.len();
            let excess = current_size.saturating_sub(self.target_vocab_size);
            let prune_target = ((current_size as f64) * self.pruning_rate).ceil() as usize;
            let num_to_prune = prune_target.min(excess);

            let mut keep_indices = HashSet::with_capacity(current_size - num_to_prune);
            for &(idx, _) in &candidates[num_to_prune..] {
                keep_indices.insert(idx);
            }
            // Always keep protected single bytes
            for &(idx, loss) in &candidates {
                if loss.is_infinite() {
                    keep_indices.insert(idx);
                }
            }

            let mut sorted_keep: Vec<usize> = keep_indices.into_iter().collect();
            sorted_keep.sort_unstable();

            let mut next_vocab = Vec::with_capacity(sorted_keep.len());
            // Put base 256 bytes first
            for b in 0..=255u8 {
                let byte_vec = vec![b];
                if let Some(pos) = current_vocab.iter().position(|(bytes, _)| bytes == &byte_vec) {
                    next_vocab.push(current_vocab[pos].clone());
                } else {
                    next_vocab.push((byte_vec, -15.0));
                }
            }
            // Add remaining kept tokens in deterministic sorted index order
            for idx in sorted_keep {
                if current_vocab[idx].0.len() > 1 {
                    next_vocab.push(current_vocab[idx].clone());
                }
            }

            current_vocab = next_vocab;
        }

        // 3. Final model assembly
        let mut final_vocab = Vocabulary::new_byte_level();
        let mut final_log_probs = Vec::with_capacity(current_vocab.len());

        for (bytes, lp) in &current_vocab {
            if bytes.len() > 1 {
                final_vocab.add_token(bytes.clone());
            }
            final_log_probs.push(*lp);
        }

        Ok(UnigramModel::new(final_vocab, final_log_probs))
    }
}
