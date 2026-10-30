use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use tokenisers_core::curricula::morpheme::{MorphemeGuidedScorer, MorphemeLexicon};
use tokenisers_core::curricula::super_bpe::{SuperBpeConfig, SuperBpeTrainer};
use tokenisers_core::engine::bottom_up::{BottomUpModel, BottomUpTrainer};
use tokenisers_core::engine::unigram::{UnigramModel, UnigramTrainer};
use tokenisers_core::export::huggingface::{HfModelConfig, HfTokenizerJson};
use tokenisers_core::inference::byte_entropy::{ByteEntropyConfig, ByteEntropyPatcher};
use tokenisers_core::inference::unigram_sample::UnigramSampler;
use tokenisers_core::scoring::{
    BpeScorer, LengthPenalizedScorer, MdlScorer, WordPieceScorer, WpBpeBaseScorer, WpBpeFtScorer,
    WpBpeNpmiScorer, WpBpePigScorer, WpBpePpigScorer, WpBpeScpScorer,
};
use tokenisers_core::vocab::{Vocabulary, BASE_VOCAB_SIZE};

fn extract_word_counts(corpus: &[String]) -> Result<Vec<(Vec<u8>, u64)>, &'static str> {
    if corpus.is_empty() {
        return Err("Corpus cannot be empty");
    }
    let mut word_counts_map: ahash::AHashMap<Vec<u8>, u64> = ahash::AHashMap::default();
    for doc in corpus {
        if doc.is_empty() {
            continue;
        }
        let segments = SuperBpeTrainer::segment_text(doc);
        for seg in segments {
            if !seg.is_empty() {
                *word_counts_map.entry(seg).or_insert(0) += 1;
            }
        }
    }
    let word_counts: Vec<(Vec<u8>, u64)> = word_counts_map.into_iter().collect();
    if word_counts.is_empty() {
        return Err("Corpus contains only empty documents");
    }
    Ok(word_counts)
}

#[pyclass]
#[derive(Clone)]
pub struct PyBpeModel {
    pub model: BottomUpModel,
}

#[pymethods]
impl PyBpeModel {
    #[pyo3(signature = (text, dropout=None, seed=None))]
    pub fn encode(
        &self,
        text: &str,
        dropout: Option<f32>,
        seed: Option<u64>,
    ) -> PyResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(p) = dropout {
            if !(0.0..=1.0).contains(&p) || p.is_nan() {
                return Err(PyValueError::new_err(
                    "dropout probability must be between 0.0 and 1.0",
                ));
            }
            if p > 0.0 {
                return Ok(self.model.encode_with_dropout(text.as_bytes(), p, seed));
            }
        }
        Ok(self.model.encode(text.as_bytes()))
    }

    #[pyo3(signature = (bytes, dropout=None, seed=None))]
    pub fn encode_bytes(
        &self,
        bytes: &[u8],
        dropout: Option<f32>,
        seed: Option<u64>,
    ) -> PyResult<Vec<u32>> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(p) = dropout {
            if !(0.0..=1.0).contains(&p) || p.is_nan() {
                return Err(PyValueError::new_err(
                    "dropout probability must be between 0.0 and 1.0",
                ));
            }
            if p > 0.0 {
                return Ok(self.model.encode_with_dropout(bytes, p, seed));
            }
        }
        Ok(self.model.encode(bytes))
    }

    pub fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        if ids.is_empty() {
            return Ok(String::new());
        }
        let raw_bytes = self
            .model
            .decode(&ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        match String::from_utf8(raw_bytes.clone()) {
            Ok(s) => Ok(s),
            Err(_) => {
                let s: String = raw_bytes.iter().map(|&b| b as char).collect();
                Ok(s)
            }
        }
    }

    pub fn export_hf_json(&self, path: &str) -> PyResult<()> {
        let hf_json = HfTokenizerJson::from_bpe_model(&self.model, None);
        hf_json
            .export_to_file(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn vocab_size(&self) -> usize {
        self.model.vocab.len()
    }

    pub fn get_vocab(&self) -> HashMap<String, u32> {
        let mut map = HashMap::with_capacity(self.model.vocab.len());
        for (id, bytes) in self.model.vocab.id_to_bytes_map().iter().enumerate() {
            let s: String = match std::str::from_utf8(bytes) {
                Ok(valid) => valid.to_string(),
                Err(_) => bytes.iter().map(|&b| b as char).collect(),
            };
            map.insert(s, id as u32);
        }
        map
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.vocab.get_id(token.as_bytes()).or_else(|| {
            let b: Vec<u8> = token.chars().map(|c| c as u8).collect();
            self.model.vocab.get_id(&b)
        })
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.vocab.get_bytes(id).map(|bytes| {
            match std::str::from_utf8(bytes) {
                Ok(valid) => valid.to_string(),
                Err(_) => bytes.iter().map(|&b| b as char).collect(),
            }
        })
    }

    pub fn id_to_bytes(&self, id: u32) -> Option<Vec<u8>> {
        self.model.vocab.get_bytes(id).map(|b| b.to_vec())
    }

    #[staticmethod]
    pub fn from_hf_json(path: &str) -> PyResult<Self> {
        let file_path = if Path::new(path).is_dir() || Path::new(path).extension().is_none() {
            Path::new(path).join("tokenizer.json")
        } else {
            Path::new(path).to_path_buf()
        };
        let file = File::open(&file_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open {path}: {e}")))?;
        let json: HfTokenizerJson = serde_json::from_reader(file)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse {path}: {e}")))?;
        let pretokenize = json.pretokenize();

        match json.model {
            HfModelConfig::BPE { vocab, merges, .. } => {
                let mut sorted_tokens: Vec<(u32, String)> =
                    vocab.into_iter().map(|(k, v)| (v, k)).collect();
                sorted_tokens.sort_by_key(|&(id, _)| id);

                let mut model_vocab = Vocabulary::new_byte_level();
                for (id, s) in sorted_tokens {
                    if id >= 256 {
                        let bytes = model_vocab
                            .printable_to_bytes(&s)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                        model_vocab.add_token(bytes);
                    }
                }

                let mut merges_vec = Vec::with_capacity(merges.len());
                for m in merges {
                    let parts: Vec<&str> = m.split(' ').collect();
                    if parts.len() == 2 {
                        let bytes_u = model_vocab
                            .printable_to_bytes(parts[0])
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                        let bytes_v = model_vocab
                            .printable_to_bytes(parts[1])
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                        if let (Some(u), Some(v)) = (
                            model_vocab.get_id(&bytes_u),
                            model_vocab.get_id(&bytes_v),
                        ) {
                            let mut merged_bytes = bytes_u;
                            merged_bytes.extend_from_slice(&bytes_v);
                            let merged = model_vocab.add_token(merged_bytes);
                            merges_vec.push((u, v, merged));
                        }
                    }
                }

                let mut model = BottomUpModel::new(model_vocab, merges_vec);
                model.pretokenize = pretokenize;
                Ok(Self { model })
            }
            _ => Err(PyValueError::new_err(
                "Exported tokenizer.json is not a BPE model",
            )),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyUnigramModel {
    pub model: UnigramModel,
}

#[pymethods]
impl PyUnigramModel {
    #[pyo3(signature = (text, temperature=None, nbest_size=None, sample=None, seed=None))]
    pub fn encode(
        &self,
        text: &str,
        temperature: Option<f64>,
        nbest_size: Option<usize>,
        sample: Option<bool>,
        seed: Option<u64>,
    ) -> PyResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(t) = temperature {
            if t < 0.0 || t.is_nan() {
                return Err(PyValueError::new_err(
                    "temperature must be non-negative",
                ));
            }
        }

        let is_sampling = sample.unwrap_or(false)
            || (temperature.is_some() && temperature != Some(0.0))
            || (nbest_size.is_some() && nbest_size.unwrap() > 1);

        if !is_sampling || temperature == Some(0.0) {
            return Ok(self.model.encode(text.as_bytes()));
        }

        let temp = temperature.unwrap_or(1.0);
        if temp < 0.01 {
            return Ok(self.model.encode(text.as_bytes()));
        }

        // Calibrated temperature scaling for lattice posterior sampling (subword regularization).
        // Since unigram log-probabilities span substantial ranges across whole words vs base bytes,
        // scaling the user temperature aligns user-space T in [0.1, 1.0] with posterior lattice entropy.
        let scaled_temp = temp * 25.0;
        let sampler = UnigramSampler::new(&self.model.trie);
        if let Some(k) = nbest_size {
            if k > 1 {
                return Ok(sampler.sample_kbest(text.as_bytes(), k, scaled_temp, seed));
            } else {
                return Ok(self.model.encode(text.as_bytes()));
            }
        }

        Ok(sampler.sample_ffbs(text.as_bytes(), scaled_temp, seed))
    }

    pub fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        if ids.is_empty() {
            return Ok(String::new());
        }
        let raw_bytes = self
            .model
            .decode(&ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        match String::from_utf8(raw_bytes.clone()) {
            Ok(s) => Ok(s),
            Err(_) => {
                let s: String = raw_bytes.iter().map(|&b| b as char).collect();
                Ok(s)
            }
        }
    }

    pub fn export_hf_json(&self, path: &str) -> PyResult<()> {
        let hf_json = HfTokenizerJson::from_unigram_model(&self.model, None);
        hf_json
            .export_to_file(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn vocab_size(&self) -> usize {
        self.model.vocab.len()
    }

    pub fn get_vocab(&self) -> HashMap<String, u32> {
        let mut map = HashMap::with_capacity(self.model.vocab.len());
        for (id, bytes) in self.model.vocab.id_to_bytes_map().iter().enumerate() {
            let s: String = match std::str::from_utf8(bytes) {
                Ok(valid) => valid.to_string(),
                Err(_) => bytes.iter().map(|&b| b as char).collect(),
            };
            map.insert(s, id as u32);
        }
        map
    }

    pub fn get_log_probs(&self) -> HashMap<String, f64> {
        let mut map = HashMap::with_capacity(self.model.vocab.len());
        let mut max_lp = f64::NEG_INFINITY;
        for &lp in &self.model.log_probs {
            if lp > max_lp {
                max_lp = lp;
            }
        }
        let sum_exp: f64 = self
            .model
            .log_probs
            .iter()
            .map(|&lp| (lp - max_lp).exp())
            .sum();
        let log_sum_exp = max_lp + sum_exp.ln();

        for (id, bytes) in self.model.vocab.id_to_bytes_map().iter().enumerate() {
            let s: String = match std::str::from_utf8(bytes) {
                Ok(valid) => valid.to_string(),
                Err(_) => bytes.iter().map(|&b| b as char).collect(),
            };
            let lp = if id < self.model.log_probs.len() {
                self.model.log_probs[id]
            } else {
                -15.0
            };
            let normalized_lp = lp - log_sum_exp;
            map.insert(s, normalized_lp);
        }
        map
    }

    #[staticmethod]
    pub fn from_hf_json(path: &str) -> PyResult<Self> {
        let file_path = if Path::new(path).is_dir() || Path::new(path).extension().is_none() {
            Path::new(path).join("tokenizer.json")
        } else {
            Path::new(path).to_path_buf()
        };
        let file = File::open(&file_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open {path}: {e}")))?;
        let json: HfTokenizerJson = serde_json::from_reader(file)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse {path}: {e}")))?;

        match json.model {
            HfModelConfig::Unigram { vocab, .. } => {
                let mut model_vocab = Vocabulary::new_byte_level();
                let mut log_probs = Vec::with_capacity(vocab.len());
                for (s, score) in vocab {
                    let bytes = model_vocab
                        .printable_to_bytes(&s)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    if bytes.len() > 1 {
                        model_vocab.add_token(bytes);
                    }
                    log_probs.push(score);
                }
                Ok(Self {
                    model: UnigramModel::new(model_vocab, log_probs),
                })
            }
            _ => Err(PyValueError::new_err(
                "Exported tokenizer.json is not a Unigram model",
            )),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyByteEntropyPatcher {
    pub patcher: ByteEntropyPatcher,
}

#[pymethods]
impl PyByteEntropyPatcher {
    #[new]
    #[pyo3(signature = (entropy_threshold=4.2, context_order=3, min_len=2, max_len=16, smoothing_alpha=0.05))]
    pub fn new(
        entropy_threshold: f64,
        context_order: usize,
        min_len: usize,
        max_len: usize,
        smoothing_alpha: f64,
    ) -> PyResult<Self> {
        if entropy_threshold <= 0.0 || entropy_threshold.is_nan() {
            return Err(PyValueError::new_err(
                "entropy_threshold must be strictly positive",
            ));
        }
        let config = ByteEntropyConfig {
            context_order,
            tau_threshold: entropy_threshold,
            min_len,
            max_len,
        };
        let mut patcher = ByteEntropyPatcher::new(config);
        patcher.smoothing_alpha = smoothing_alpha;
        Ok(Self { patcher })
    }

    pub fn segment_stream(&self, bytes: &[u8]) -> Vec<Vec<u8>> {
        self.patcher.segment_stream(bytes)
    }

    pub fn train_transitions(&mut self, corpus_bytes: &[u8]) {
        self.patcher.train_transitions(corpus_bytes);
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (corpus, vocab_size, scorer_type="frequency", min_frequency=0, alpha=1.0, beta=1.0, penalty=0.0, special_tokens=None))]
pub fn train_bpe(
    py: Python<'_>,
    corpus: Vec<String>,
    vocab_size: usize,
    scorer_type: &str,
    min_frequency: u64,
    alpha: f64,
    beta: f64,
    penalty: f64,
    special_tokens: Option<Vec<String>>,
) -> PyResult<PyBpeModel> {
    if vocab_size < BASE_VOCAB_SIZE {
        return Err(PyValueError::new_err(format!(
            "vocab_size must be at least {BASE_VOCAB_SIZE}"
        )));
    }
    if beta.is_nan() {
        return Err(PyValueError::new_err("beta must be a valid float"));
    }
    let scorer_norm = scorer_type.to_lowercase();
    match scorer_norm.as_str() {
        "frequency" | "bpe" | "wordpiece" | "pmi" | "mdl" | "length_penalized"
        | "wp-bpe-base" | "wp_bpe_base"
        | "wp-bpe-ft" | "wp_bpe_ft"
        | "wp-bpe-pig" | "wp_bpe_pig"
        | "wp-bpe-scp" | "wp_bpe_scp"
        | "wp-bpe-npmi" | "wp_bpe_npmi"
        | "wp-bpe-ppig" | "wp_bpe_ppig" => {}
        unknown => {
            return Err(PyValueError::new_err(format!(
                "Unknown scorer type: {unknown}"
            )));
        }
    }

    let min_freq = min_frequency.max(1);

    let model = py.allow_threads(|| -> Result<BottomUpModel, String> {
        let word_counts = extract_word_counts(&corpus).map_err(|e| e.to_string())?;

        let mut model = match scorer_norm.as_str() {
            "frequency" | "bpe" => {
                let scorer = BpeScorer::new();
                BottomUpTrainer::new(&scorer, vocab_size)
                    .with_min_frequency(min_freq)
                    .train(&word_counts)
                    .map_err(|e| e.to_string())?
            }
            "wordpiece" | "pmi" => {
                let scorer = WordPieceScorer::new(alpha, min_freq);
                BottomUpTrainer::new(&scorer, vocab_size)
                    .with_min_frequency(min_freq)
                    .train(&word_counts)
                    .map_err(|e| e.to_string())?
            }
            "mdl" => {
                let entry_cost = if penalty > 0.0 { penalty } else { 24.0 };
                let scorer = MdlScorer::new(entry_cost, min_freq);
                BottomUpTrainer::new(&scorer, vocab_size)
                    .with_min_frequency(min_freq)
                    .train(&word_counts)
                    .map_err(|e| e.to_string())?
            }
            "length_penalized" => {
                let scorer = LengthPenalizedScorer::new(beta, min_freq);
                BottomUpTrainer::new(&scorer, vocab_size)
                    .with_min_frequency(min_freq)
                    .train(&word_counts)
                    .map_err(|e| e.to_string())?
            }
            "wp-bpe-base" | "wp_bpe_base" => {
                let scorer = WpBpeBaseScorer::new();
                BottomUpTrainer::new(&scorer, vocab_size)
                    .with_min_frequency(1)
                    .train(&word_counts)
                    .map_err(|e| e.to_string())?
            }
            "wp-bpe-ft" | "wp_bpe_ft" => {
                let scorer = WpBpeFtScorer::new(min_freq.max(2));
                BottomUpTrainer::new(&scorer, vocab_size)
                    .with_min_frequency(min_freq.max(2))
                    .train(&word_counts)
                    .map_err(|e| e.to_string())?
            }
            "wp-bpe-pig" | "wp_bpe_pig" => {
                let scorer = WpBpePigScorer::new(min_freq);
                BottomUpTrainer::new(&scorer, vocab_size)
                    .with_min_frequency(min_freq)
                    .train(&word_counts)
                    .map_err(|e| e.to_string())?
            }
            "wp-bpe-scp" | "wp_bpe_scp" => {
                let discount = if alpha != 1.0 { alpha } else { 0.75 };
                let scorer = WpBpeScpScorer::new(min_freq.max(2), discount);
                BottomUpTrainer::new(&scorer, vocab_size)
                    .with_min_frequency(min_freq.max(2))
                    .train(&word_counts)
                    .map_err(|e| e.to_string())?
            }
            "wp-bpe-npmi" | "wp_bpe_npmi" => {
                let alpha_val = if alpha != 1.0 { alpha } else { 0.5 };
                let scorer = WpBpeNpmiScorer::new(min_freq.max(2), alpha_val);
                BottomUpTrainer::new(&scorer, vocab_size)
                    .with_min_frequency(min_freq.max(2))
                    .train(&word_counts)
                    .map_err(|e| e.to_string())?
            }
            "wp-bpe-ppig" | "wp_bpe_ppig" => {
                let len_penalty = if penalty > 0.0 { penalty } else { 0.35 };
                let scorer = WpBpePpigScorer::new(min_freq.max(2), len_penalty);
                BottomUpTrainer::new(&scorer, vocab_size)
                    .with_min_frequency(min_freq.max(2))
                    .train(&word_counts)
                    .map_err(|e| e.to_string())?
            }
            _ => unreachable!(),
        };

        if let Some(specials) = &special_tokens {
            for special in specials {
                model
                    .vocab
                    .add_special_token(special)
                    .map_err(|e| e.to_string())?;
            }
        }

        let mut pad_idx = 0;
        while model.vocab.len() < vocab_size {
            let token_str = format!("<unused_{pad_idx}>");
            model.vocab.add_token(token_str.into_bytes());
            pad_idx += 1;
        }

        Ok(model)
    }).map_err(PyValueError::new_err)?;

    Ok(PyBpeModel { model })
}

#[pyfunction]
#[pyo3(signature = (corpus, vocab_size, num_iterations=10, shrinking_factor=0.75))]
pub fn train_unigram(
    py: Python<'_>,
    corpus: Vec<String>,
    vocab_size: usize,
    num_iterations: usize,
    shrinking_factor: f64,
) -> PyResult<PyUnigramModel> {
    if vocab_size < BASE_VOCAB_SIZE {
        return Err(PyValueError::new_err(format!(
            "vocab_size must be at least {BASE_VOCAB_SIZE}"
        )));
    }

    let pruning_rate = (1.0 - shrinking_factor).clamp(0.01, 0.50);
    let mut trainer = UnigramTrainer::new(vocab_size, pruning_rate);
    trainer.em_sub_iterations = num_iterations.clamp(1, 10);

    let model = py.allow_threads(|| -> Result<UnigramModel, String> {
        let word_counts = extract_word_counts(&corpus).map_err(|e| e.to_string())?;
        trainer
            .train(&word_counts)
            .map_err(|e| e.to_string())
    }).map_err(PyValueError::new_err)?;

    Ok(PyUnigramModel { model })
}

#[pyfunction]
#[pyo3(signature = (corpus, v_switch, v_final))]
pub fn train_super_bpe(
    py: Python<'_>,
    corpus: Vec<String>,
    v_switch: usize,
    v_final: usize,
) -> PyResult<PyBpeModel> {
    if corpus.is_empty() {
        return Err(PyValueError::new_err("Corpus cannot be empty"));
    }
    if v_switch < BASE_VOCAB_SIZE {
        return Err(PyValueError::new_err(format!(
            "v_switch must be at least {BASE_VOCAB_SIZE}"
        )));
    }
    if v_switch > v_final {
        return Err(PyValueError::new_err(format!(
            "v_switch ({v_switch}) cannot exceed v_final ({v_final})"
        )));
    }

    let config = SuperBpeConfig::new(v_switch, v_final)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let model = py.allow_threads(|| -> Result<BottomUpModel, String> {
        let scorer = BpeScorer::new();
        let trainer = SuperBpeTrainer::new(config, &scorer);
        let line_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        trainer
            .train(&line_refs)
            .map_err(|e| e.to_string())
    }).map_err(PyValueError::new_err)?;

    Ok(PyBpeModel { model })
}

/// Morpheme-guided BPE. `morpheme_lexicon` maps training words (whole whitespace-free segments)
/// to their surface morphemes; it must come from training data or an unsupervised segmenter, never
/// from evaluation gold. Inner-scorer parameters left as `None` take the scorer's defaults.
#[pyfunction]
#[pyo3(signature = (corpus, vocab_size, morpheme_lexicon, lambda_weight=1.0, scorer="frequency", min_frequency=1, alpha=None, beta=None, penalty=None, special_tokens=None))]
#[allow(clippy::too_many_arguments)]
pub fn train_morpheme_guided(
    py: Python<'_>,
    corpus: Vec<String>,
    vocab_size: usize,
    morpheme_lexicon: HashMap<String, Vec<String>>,
    lambda_weight: f64,
    scorer: &str,
    min_frequency: u64,
    alpha: Option<f64>,
    beta: Option<f64>,
    penalty: Option<f64>,
    special_tokens: Option<Vec<String>>,
) -> PyResult<PyBpeModel> {
    let specials = special_tokens.unwrap_or_default();
    if vocab_size < BASE_VOCAB_SIZE + specials.len() {
        return Err(PyValueError::new_err(format!(
            "vocab_size must be at least {BASE_VOCAB_SIZE} + the number of special tokens ({})",
            specials.len()
        )));
    }
    if !lambda_weight.is_finite() || lambda_weight < 0.0 {
        return Err(PyValueError::new_err(format!(
            "lambda_weight must be a finite number >= 0, got {lambda_weight}"
        )));
    }

    let scorer_norm = scorer.to_lowercase();
    let applies_to = match scorer_norm.as_str() {
        "frequency" | "bpe" => "",
        "wordpiece" | "pmi" => "alpha",
        "mdl" => "penalty",
        "length_penalized" => "beta",
        unknown => {
            return Err(PyValueError::new_err(format!(
                "Unknown scorer type for morpheme-guided BPE: {unknown}"
            )));
        }
    };
    for (name, value) in [("alpha", alpha), ("beta", beta), ("penalty", penalty)] {
        let Some(value) = value else { continue };
        if name != applies_to {
            return Err(PyValueError::new_err(format!(
                "{name} does not apply to scorer '{scorer}'"
            )));
        }
        if !value.is_finite() || (name != "beta" && value < 0.0) {
            return Err(PyValueError::new_err(format!("invalid {name}: {value}")));
        }
    }

    let mut lexicon = MorphemeLexicon::new();
    let mut rejected = Vec::new();
    for (word, morphemes) in &morpheme_lexicon {
        let m_refs: Vec<&[u8]> = morphemes.iter().map(|m| m.as_bytes()).collect();
        if !lexicon.insert_segmentation(word.as_bytes(), &m_refs) {
            rejected.push(word.as_str());
        }
    }
    if !rejected.is_empty() {
        rejected.sort_unstable();
        let msg = format!(
            "morpheme_lexicon: ignored {} entries whose morphemes do not concatenate to the word \
             (byte offsets need surface segmentations): {:?}",
            rejected.len(),
            &rejected[..rejected.len().min(10)]
        );
        let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(py, category.as_any(), &std::ffi::CString::new(msg)?, 1)?;
    }

    let min_freq = min_frequency.max(1);
    let target = vocab_size - specials.len();
    let model = py.allow_threads(|| -> Result<BottomUpModel, String> {
        let word_counts = extract_word_counts(&corpus).map_err(|e| e.to_string())?;

        macro_rules! train_with_inner {
            ($inner_scorer:expr) => {{
                let morpheme_scorer =
                    MorphemeGuidedScorer::new($inner_scorer, lexicon, lambda_weight, 1.0);
                BottomUpTrainer::new(&morpheme_scorer, target)
                    .with_min_frequency(min_freq)
                    .train(&word_counts)
                    .map_err(|e| e.to_string())?
            }};
        }

        let mut model = match scorer_norm.as_str() {
            "frequency" | "bpe" => train_with_inner!(BpeScorer::new()),
            "wordpiece" | "pmi" => train_with_inner!(WordPieceScorer::new(
                alpha.unwrap_or(WordPieceScorer::default().alpha),
                min_freq
            )),
            "mdl" => train_with_inner!(MdlScorer::new(
                penalty.unwrap_or(MdlScorer::default().entry_overhead_bits),
                min_freq
            )),
            "length_penalized" => train_with_inner!(LengthPenalizedScorer::new(
                beta.unwrap_or(LengthPenalizedScorer::default().beta),
                min_freq
            )),
            _ => unreachable!(),
        };

        for special in &specials {
            model
                .vocab
                .add_special_token(special)
                .map_err(|e| e.to_string())?;
        }
        Ok(model)
    }).map_err(PyValueError::new_err)?;

    // No <unused_N> padding: a strong prior (or a small corpus) can leave fewer mergeable pairs.
    if model.vocab.len() < vocab_size {
        let msg = format!(
            "morpheme-guided training stopped at {} tokens (requested {vocab_size}): no eligible merges left",
            model.vocab.len()
        );
        let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(py, category.as_any(), &std::ffi::CString::new(msg)?, 1)?;
    }

    Ok(PyBpeModel { model })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyBpeModel>()?;
    m.add_class::<PyUnigramModel>()?;
    m.add_class::<PyByteEntropyPatcher>()?;
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(train_unigram, m)?)?;
    m.add_function(wrap_pyfunction!(train_super_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(train_morpheme_guided, m)?)?;
    Ok(())
}
