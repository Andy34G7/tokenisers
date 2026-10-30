use ahash::AHashMap;

/// Configuration for Byte-Entropy Dynamic Patching.
#[derive(Debug, Clone)]
pub struct ByteEntropyConfig {
    /// Markov context order in bytes (e.g. 2, 3, or 4).
    pub context_order: usize,
    /// Next-byte entropy threshold in bits (e.g. 3.5 to 5.0): a new patch starts before a byte
    /// whose predictive entropy H(next | history) exceeds it.
    pub tau_threshold: f64,
    /// Minimum patch length in bytes (e.g. 2).
    pub min_len: usize,
    /// Maximum patch length in bytes (e.g. 16 or 32).
    pub max_len: usize,
}

impl Default for ByteEntropyConfig {
    fn default() -> Self {
        Self {
            context_order: 3,
            tau_threshold: 4.2,
            min_len: 2,
            max_len: 16,
        }
    }
}

/// BLT-style dynamic byte patcher driven by the predictive entropy of a backoff Markov model.
#[derive(Debug, Clone)]
pub struct ByteEntropyPatcher {
    pub config: ByteEntropyConfig,
    /// Next-byte counts `(byte, count)` for every context of length 0..=context_order seen in
    /// training, sparse: a dense `[u32; 256]` per context cost ~1 KB each (GBs on large corpora).
    pub transition_table: AHashMap<Vec<u8>, Vec<(u8, u32)>>,
    /// Additive Laplace smoothing constant.
    pub smoothing_alpha: f64,
}

impl Default for ByteEntropyPatcher {
    fn default() -> Self {
        Self::new(ByteEntropyConfig::default())
    }
}

impl ByteEntropyPatcher {
    pub fn new(config: ByteEntropyConfig) -> Self {
        Self {
            config,
            transition_table: AHashMap::default(),
            smoothing_alpha: 0.05,
        }
    }

    /// Trains the Markov model on a raw byte corpus: for every byte, counts it after each of
    /// its preceding contexts of length 0..=context_order (shorter ones serve as backoff).
    pub fn train_transitions(&mut self, corpus_bytes: &[u8]) {
        let order = self.config.context_order;
        for (i, &next_byte) in corpus_bytes.iter().enumerate() {
            for k in 0..=order.min(i) {
                let ctx = &corpus_bytes[i - k..i];
                if !self.transition_table.contains_key(ctx) {
                    self.transition_table.insert(ctx.to_vec(), Vec::new());
                }
                let counts = self.transition_table.get_mut(ctx).expect("inserted above");
                match counts.iter_mut().find(|(b, _)| *b == next_byte) {
                    Some((_, c)) => *c += 1,
                    None => counts.push((next_byte, 1)),
                }
            }
        }
    }

    /// True once `train_transitions` has seen at least one byte. An untrained patcher predicts
    /// 8 bits everywhere, i.e. cuts a patch every `min_len` bytes whenever `tau < 8`.
    pub fn is_trained(&self) -> bool {
        !self.transition_table.is_empty()
    }

    /// Entropy (bits) of the next byte given everything before it, using the longest suffix of
    /// `history` (at most `context_order` bytes) seen in training; 8.0 if none was seen.
    pub fn next_byte_entropy(&self, history: &[u8]) -> f64 {
        let longest = self.config.context_order.min(history.len());
        (0..=longest)
            .rev()
            .map(|k| &history[history.len() - k..])
            .find(|ctx| self.transition_table.contains_key(*ctx))
            .map_or(8.0, |ctx| self.compute_entropy(ctx))
    }

    /// Predictive entropy H(B | c) = -sum P log2 P in bits for the exact context `c` (Laplace
    /// smoothed); 8.0 for a context never seen in training.
    pub fn compute_entropy(&self, context: &[u8]) -> f64 {
        let Some(counts) = self.transition_table.get(context) else {
            return 8.0;
        };
        let alpha = self.smoothing_alpha;
        let total: u64 = counts.iter().map(|&(_, c)| c as u64).sum();
        let denom = total as f64 + 256.0 * alpha;
        let term = |c: f64| {
            let p = (c + alpha) / denom;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        };
        // Bytes absent from the sparse list have count 0.
        counts.iter().map(|&(_, c)| term(c as f64)).sum::<f64>()
            + (256 - counts.len()) as f64 * term(0.0)
    }

    /// Segments a byte stream into patches (lossless: their concatenation is `bytes`).
    ///
    /// A new patch starts before byte `i` when `next_byte_entropy(&bytes[..i]) > tau_threshold`
    /// (BLT global-threshold rule: no look-ahead at byte `i` itself), subject to the guardrails
    /// `min_len <= patch length <= max_len` (the final patch may be shorter than `min_len`).
    pub fn segment_stream(&self, bytes: &[u8]) -> Vec<Vec<u8>> {
        if bytes.is_empty() {
            return Vec::new();
        }

        let mut patches = Vec::new();
        let mut curr_patch = Vec::new();

        for (idx, &byte) in bytes.iter().enumerate() {
            curr_patch.push(byte);
            let patch_len = curr_patch.len();

            // Guardrail 1: Minimum length constraint suppresses boundary
            if patch_len < self.config.min_len {
                continue;
            }

            // Guardrail 2: Maximum length constraint forces boundary
            if patch_len >= self.config.max_len {
                patches.push(curr_patch);
                curr_patch = Vec::new();
                continue;
            }

            // Entropy rule: cut before the next byte if it is hard to predict.
            if idx + 1 < bytes.len()
                && self.next_byte_entropy(&bytes[..=idx]) > self.config.tau_threshold
            {
                patches.push(curr_patch);
                curr_patch = Vec::new();
            }
        }

        if !curr_patch.is_empty() {
            patches.push(curr_patch);
        }

        patches
    }
}
