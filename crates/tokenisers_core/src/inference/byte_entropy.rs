use ahash::AHashMap;

/// Configuration for Byte-Entropy Dynamic Patching.
#[derive(Debug, Clone)]
pub struct ByteEntropyConfig {
    /// Markov context order in bytes (e.g. 2, 3, or 4).
    pub context_order: usize,
    /// Entropy / surprise threshold in bits (e.g. 3.5 to 5.0).
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

/// Dynamic byte stream patcher based on Markov transition surprise and predictive entropy.
#[derive(Debug, Clone)]
pub struct ByteEntropyPatcher {
    pub config: ByteEntropyConfig,
    /// Markov transition counts: context byte sequence -> 256 byte frequency counts.
    transition_table: AHashMap<Vec<u8>, [u32; 256]>,
    /// Total transitions observed per context.
    context_totals: AHashMap<Vec<u8>, u32>,
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
            context_totals: AHashMap::default(),
            smoothing_alpha: 0.05,
        }
    }

    /// Trains the Markov transition table on raw byte corpus.
    pub fn train_transitions(&mut self, corpus_bytes: &[u8]) {
        let order = self.config.context_order;
        if corpus_bytes.len() <= order {
            return;
        }

        for i in 0..(corpus_bytes.len() - order) {
            let ctx = corpus_bytes[i..(i + order)].to_vec();
            let next_byte = corpus_bytes[i + order];

            let counts = self.transition_table.entry(ctx.clone()).or_insert([0; 256]);
            counts[next_byte as usize] += 1;
            *self.context_totals.entry(ctx).or_insert(0) += 1;
        }
    }

    /// Computes transition surprise S(b | c) = -log2 P(b | c) in bits.
    pub fn compute_surprise(&self, context: &[u8], next_byte: u8) -> f64 {
        let alpha = self.smoothing_alpha;
        let v = 256.0;

        if let Some(counts) = self.transition_table.get(context) {
            let total = self.context_totals[context] as f64;
            let count = counts[next_byte as usize] as f64;
            let prob = (count + alpha) / (total + v * alpha);
            -prob.log2()
        } else {
            // Unseen context: uniform prior
            8.0 // -log2(1/256) = 8.0 bits
        }
    }

    /// Computes predictive conditional entropy H(B | c) = -sum P log2 P in bits.
    pub fn compute_entropy(&self, context: &[u8]) -> f64 {
        let alpha = self.smoothing_alpha;
        let v = 256.0;

        if let Some(counts) = self.transition_table.get(context) {
            let total = self.context_totals[context] as f64;
            let denom = total + v * alpha;
            let mut h = 0.0;
            for &c in counts.iter() {
                let p = (c as f64 + alpha) / denom;
                if p > 0.0 {
                    h -= p * p.log2();
                }
            }
            h
        } else {
            8.0
        }
    }

    /// Segments an input byte stream into dynamic patches with [min_len, max_len] guardrails.
    pub fn segment_stream(&self, bytes: &[u8]) -> Vec<Vec<u8>> {
        if bytes.is_empty() {
            return Vec::new();
        }

        let mut patches = Vec::new();
        let mut curr_patch = Vec::new();
        let order = self.config.context_order;

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

            // Guardrail 3: Evaluate transition surprise at boundary
            if idx + 1 < bytes.len() {
                let next_byte = bytes[idx + 1];
                let ctx_start = (idx + 1).saturating_sub(order);
                let ctx = &bytes[ctx_start..=idx];

                let surprise = self.compute_surprise(ctx, next_byte);
                if surprise >= self.config.tau_threshold {
                    patches.push(curr_patch);
                    curr_patch = Vec::new();
                }
            }
        }

        if !curr_patch.is_empty() {
            patches.push(curr_patch);
        }

        patches
    }
}
