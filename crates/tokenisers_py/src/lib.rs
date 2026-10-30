use std::collections::{BTreeMap, HashMap};
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
    BpeScorer, LengthPenalizedScorer, MdlScorer, PairScorer, WordPieceScorer, WpBpeBaseScorer,
    WpBpeFtScorer, WpBpeNpmiScorer, WpBpePigScorer, WpBpePpigScorer, WpBpeScpScorer,
};
use tokenisers_core::vocab::{TokenId, Vocabulary, BASE_VOCAB_SIZE};

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
    let mut word_counts: Vec<(Vec<u8>, u64)> = word_counts_map.into_iter().collect();
    if word_counts.is_empty() {
        return Err("Corpus contains only empty documents");
    }
    // AHashMap iteration order is seeded per process: sort so training input is reproducible.
    word_counts.sort_unstable();
    Ok(word_counts)
}

fn value_err(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The string a token is known by in `get_vocab`/`id_to_token`/`token_to_id`: a special token's
/// raw content, otherwise the GPT-2 byte-level printable form of its bytes. These are the keys the
/// HF export writes, and they are unique per id (raw latin-1/UTF-8 strings are not).
fn token_string(vocab: &Vocabulary, id: TokenId) -> Option<String> {
    let bytes = vocab.get_bytes(id)?;
    Some(match vocab.special_tokens().iter().find(|s| s.id == id) {
        Some(special) => special.content.clone(),
        None => vocab.bytes_to_printable(bytes),
    })
}

fn token_id(vocab: &Vocabulary, token: &str) -> Option<TokenId> {
    if let Some(special) = vocab.special_tokens().iter().find(|s| s.content == token) {
        return Some(special.id);
    }
    vocab
        .printable_to_bytes(token)
        .ok()
        .and_then(|bytes| vocab.get_id(&bytes))
}

fn vocab_map(vocab: &Vocabulary) -> HashMap<String, u32> {
    (0..vocab.len() as TokenId)
        .filter_map(|id| token_string(vocab, id).map(|s| (s, id)))
        .collect()
}

fn hf_json_path(path: &str) -> std::path::PathBuf {
    let p = Path::new(path);
    if p.is_dir() || p.extension().is_none() {
        p.join("tokenizer.json")
    } else {
        p.to_path_buf()
    }
}

fn read_hf_json(path: &str) -> PyResult<HfTokenizerJson> {
    let file = File::open(hf_json_path(path))
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open {path}: {e}")))?;
    serde_json::from_reader(file)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse {path}: {e}")))
}

#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyBpeModel {
    pub model: BottomUpModel,
}

#[pymethods]
impl PyBpeModel {
    pub fn vocab_size(&self) -> usize {
        self.model.vocab.len()
    }

    /// `{token string: id}` with exactly `vocab_size()` entries (see `token_string`).
    pub fn get_vocab(&self) -> HashMap<String, u32> {
        vocab_map(&self.model.vocab)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        token_id(&self.model.vocab, token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        token_string(&self.model.vocab, id)
    }

    pub fn id_to_bytes(&self, id: u32) -> Option<Vec<u8>> {
        self.model.vocab.get_bytes(id).map(|b| b.to_vec())
    }

    /// Raw bytes of the ids; `ValueError` on an unknown id.
    pub fn decode_bytes(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<Vec<u8>> {
        py.allow_threads(|| self.model.vocab.decode(&ids)).map_err(value_err)
    }

    /// UTF-8 text of the ids, invalid sequences replaced by U+FFFD.
    pub fn decode(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<String> {
        let bytes = self.decode_bytes(py, ids)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    #[pyo3(signature = (text, dropout=None, seed=None))]
    pub fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        dropout: Option<f32>,
        seed: Option<u64>,
    ) -> PyResult<Vec<u32>> {
        self.encode_bytes(py, text.as_bytes(), dropout, seed)
    }

    /// BPE (with BPE-dropout probability `dropout`, HF semantics) of raw bytes. Releases the GIL.
    #[pyo3(signature = (bytes, dropout=None, seed=None))]
    pub fn encode_bytes(
        &self,
        py: Python<'_>,
        bytes: &[u8],
        dropout: Option<f32>,
        seed: Option<u64>,
    ) -> PyResult<Vec<u32>> {
        let p = dropout.unwrap_or(0.0);
        if !(0.0..=1.0).contains(&p) {
            return Err(PyValueError::new_err(
                "dropout probability must be between 0.0 and 1.0",
            ));
        }
        Ok(py.allow_threads(|| self.model.encode_with_dropout(bytes, p, seed)))
    }

    pub fn export_hf_json(&self, path: &str) -> PyResult<()> {
        let hf_json = HfTokenizerJson::from_bpe_model(&self.model, None);
        hf_json
            .export_to_file(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Loads a byte-level BPE tokenizer.json (as `export_hf_json` writes it). Ids are taken from
    /// the file: ids 0..=255 must be the 256 single bytes, the rest contiguous; special tokens come
    /// from `added_tokens`. Anything else raises `ValueError`.
    #[staticmethod]
    pub fn from_hf_json(path: &str) -> PyResult<Self> {
        let json = read_hf_json(path)?;
        let pretokenize = json.pretokenize();
        let HfModelConfig::BPE { vocab, merges, .. } = json.model else {
            return Err(PyValueError::new_err("tokenizer.json is not a BPE model"));
        };

        let specials: HashMap<u32, String> = json
            .added_tokens
            .into_iter()
            .map(|t| (t.id, t.content))
            .collect();
        let mut keys: BTreeMap<u32, String> = BTreeMap::new();
        for (key, id) in vocab {
            // A special token may also keep its byte-level key (export keeps it when merges
            // produce it); any other id must have exactly one key.
            if let Some(old) = keys.insert(id, key) {
                if !specials.contains_key(&id) {
                    return Err(value_err(format!("tokenizer.json maps both {old:?} and another token to id {id}")));
                }
            }
        }
        let n = keys.keys().chain(specials.keys()).max().map_or(0, |&m| m as usize + 1);

        let mut model_vocab = Vocabulary::new_byte_level();
        for id in 0..n as u32 {
            let got = if let Some(content) = specials.get(&id) {
                model_vocab.add_special_token(content).map_err(value_err)?
            } else {
                let key = keys
                    .get(&id)
                    .ok_or_else(|| value_err(format!("tokenizer.json has no token with id {id}")))?;
                let bytes = model_vocab.printable_to_bytes(key).map_err(value_err)?;
                if id < BASE_VOCAB_SIZE as u32 {
                    if bytes != [id as u8] {
                        return Err(value_err(format!(
                            "not a byte-level vocabulary: id {id} is {key:?}, expected byte {id}"
                        )));
                    }
                    id
                } else {
                    model_vocab.add_token(bytes)
                }
            };
            if got != id {
                return Err(value_err(format!(
                    "token with id {id} duplicates the bytes of id {got}"
                )));
            }
        }

        let mut merges_vec = Vec::with_capacity(merges.len());
        for m in &merges {
            let (u, v) = m
                .split_once(' ')
                .ok_or_else(|| value_err(format!("malformed merge {m:?}")))?;
            let lookup = |s: &str| -> PyResult<(TokenId, Vec<u8>)> {
                let bytes = model_vocab.printable_to_bytes(s).map_err(value_err)?;
                let id = model_vocab
                    .get_id(&bytes)
                    .ok_or_else(|| value_err(format!("merge {m:?}: {s:?} is not in the vocabulary")))?;
                Ok((id, bytes))
            };
            let (u_id, mut merged) = lookup(u)?;
            let (v_id, v_bytes) = lookup(v)?;
            merged.extend_from_slice(&v_bytes);
            let w = model_vocab
                .get_id(&merged)
                .ok_or_else(|| value_err(format!("merge {m:?}: result is not in the vocabulary")))?;
            merges_vec.push((u_id, v_id, w));
        }

        let mut model = BottomUpModel::new(model_vocab, merges_vec);
        model.pretokenize = pretokenize;
        Ok(Self { model })
    }
}

#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyUnigramModel {
    pub model: UnigramModel,
}

#[pymethods]
impl PyUnigramModel {
    pub fn vocab_size(&self) -> usize {
        self.model.vocab.len()
    }

    /// `{token string: id}` with exactly `vocab_size()` entries (see `token_string`).
    pub fn get_vocab(&self) -> HashMap<String, u32> {
        vocab_map(&self.model.vocab)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        token_id(&self.model.vocab, token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        token_string(&self.model.vocab, id)
    }

    pub fn id_to_bytes(&self, id: u32) -> Option<Vec<u8>> {
        self.model.vocab.get_bytes(id).map(|b| b.to_vec())
    }

    /// Raw bytes of the ids; `ValueError` on an unknown id.
    pub fn decode_bytes(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<Vec<u8>> {
        py.allow_threads(|| self.model.vocab.decode(&ids)).map_err(value_err)
    }

    /// UTF-8 text of the ids, invalid sequences replaced by U+FFFD.
    pub fn decode(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<String> {
        let bytes = self.decode_bytes(py, ids)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    #[pyo3(signature = (text, temperature=None, nbest_size=None, sample=None, seed=None))]
    pub fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        temperature: Option<f64>,
        nbest_size: Option<i64>,
        sample: Option<bool>,
        seed: Option<u64>,
    ) -> PyResult<Vec<u32>> {
        self.encode_bytes(py, text.as_bytes(), temperature, nbest_size, sample, seed)
    }

    /// Viterbi, or a sample (subword regularization) when `sample` is true. `sample` defaults to
    /// true when `temperature` is given or `nbest_size` is > 1 or < 0. A sample is drawn with
    /// probability proportional to P(s)^(1/temperature) (temperature 1 = exact posterior,
    /// SentencePiece alpha = 1/temperature, temperature 0 = Viterbi) over all segmentations
    /// (`nbest_size` None or < 0, FFBS) or over the `nbest_size` best ones; `nbest_size` 0 or 1
    /// is Viterbi (SentencePiece convention). Releases the GIL.
    #[pyo3(signature = (bytes, temperature=None, nbest_size=None, sample=None, seed=None))]
    pub fn encode_bytes(
        &self,
        py: Python<'_>,
        bytes: &[u8],
        temperature: Option<f64>,
        nbest_size: Option<i64>,
        sample: Option<bool>,
        seed: Option<u64>,
    ) -> PyResult<Vec<u32>> {
        let temp = temperature.unwrap_or(1.0);
        if temp.is_nan() || temp < 0.0 {
            return Err(PyValueError::new_err("temperature must be non-negative"));
        }
        let sample = sample.unwrap_or_else(|| {
            temperature.is_some() || nbest_size.is_some_and(|k| !(0..=1).contains(&k))
        });
        let model = &self.model;
        Ok(py.allow_threads(|| {
            let sampler = UnigramSampler::new(&model.trie);
            match nbest_size {
                _ if !sample || temp == 0.0 => model.encode(bytes),
                Some(0 | 1) => model.encode(bytes),
                Some(k) if k > 1 => sampler.sample_kbest(bytes, k as usize, temp, seed),
                _ => sampler.sample_ffbs(bytes, temp, seed),
            }
        }))
    }

    pub fn export_hf_json(&self, path: &str) -> PyResult<()> {
        let hf_json = HfTokenizerJson::from_unigram_model(&self.model, None);
        hf_json
            .export_to_file(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn get_log_probs(&self) -> HashMap<String, f64> {
        let max_lp = self
            .model
            .log_probs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = self
            .model
            .log_probs
            .iter()
            .map(|&lp| (lp - max_lp).exp())
            .sum();
        let log_sum_exp = max_lp + sum_exp.ln();
        (0..self.model.vocab.len() as TokenId)
            .filter_map(|id| {
                let lp = self.model.log_probs.get(id as usize)?;
                Some((token_string(&self.model.vocab, id)?, lp - log_sum_exp))
            })
            .collect()
    }

    /// Loads a byte-level Unigram tokenizer.json (as `export_hf_json` writes it). Entry `i` of
    /// the vocab list gets id `i`: the first 256 must be the single bytes 0..=255, the rest
    /// distinct multi-byte tokens. Anything else (or added tokens) raises `ValueError`.
    #[staticmethod]
    pub fn from_hf_json(path: &str) -> PyResult<Self> {
        let json = read_hf_json(path)?;
        let HfModelConfig::Unigram { vocab, .. } = json.model else {
            return Err(PyValueError::new_err("tokenizer.json is not a Unigram model"));
        };
        if !json.added_tokens.is_empty() {
            return Err(value_err("Unigram models with added/special tokens are not supported"));
        }
        if vocab.len() < BASE_VOCAB_SIZE {
            return Err(value_err("not a byte-level vocabulary: fewer than 256 tokens"));
        }
        let mut model_vocab = Vocabulary::new_byte_level();
        let mut log_probs = Vec::with_capacity(vocab.len());
        for (id, (key, score)) in vocab.into_iter().enumerate() {
            let bytes = model_vocab.printable_to_bytes(&key).map_err(value_err)?;
            let got = if id < BASE_VOCAB_SIZE {
                if bytes != [id as u8] {
                    return Err(value_err(format!(
                        "not a byte-level vocabulary: id {id} is {key:?}, expected byte {id}"
                    )));
                }
                id
            } else {
                model_vocab.add_token(bytes) as usize
            };
            if got != id {
                return Err(value_err(format!(
                    "token with id {id} duplicates the bytes of id {got}"
                )));
            }
            log_probs.push(score);
        }
        Ok(Self {
            model: UnigramModel::new(model_vocab, log_probs),
        })
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
        if entropy_threshold.is_nan() || entropy_threshold <= 0.0 {
            return Err(PyValueError::new_err(
                "entropy_threshold must be strictly positive",
            ));
        }
        // Training stores up to context_order + 1 contexts per corpus byte.
        if context_order > 8 {
            return Err(PyValueError::new_err(format!(
                "context_order must be at most 8, got {context_order}"
            )));
        }
        if min_len == 0 || max_len < min_len {
            return Err(PyValueError::new_err(format!(
                "need 1 <= min_len <= max_len, got min_len={min_len}, max_len={max_len}"
            )));
        }
        if !smoothing_alpha.is_finite() || smoothing_alpha < 0.0 {
            return Err(PyValueError::new_err(
                "smoothing_alpha must be a finite non-negative number",
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

    pub fn segment_stream(&self, py: Python<'_>, bytes: &[u8]) -> Vec<Vec<u8>> {
        py.allow_threads(|| self.patcher.segment_stream(bytes))
    }

    pub fn train_transitions(&mut self, corpus_bytes: &[u8]) {
        self.patcher.train_transitions(corpus_bytes);
    }

    pub fn is_trained(&self) -> bool {
        self.patcher.is_trained()
    }

    /// Sparse next-byte counts `[(context, [(byte, count), ...]), ...]`, sorted by context.
    pub fn transitions(&self) -> Vec<(Vec<u8>, Vec<(u8, u32)>)> {
        let mut out: Vec<(Vec<u8>, Vec<(u8, u32)>)> = self
            .patcher
            .transition_table
            .iter()
            .map(|(ctx, counts)| {
                let mut nonzero: Vec<(u8, u32)> =
                    counts.iter().copied().filter(|&(_, c)| c > 0).collect();
                nonzero.sort_unstable();
                (ctx.clone(), nonzero)
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Inverse of `transitions`: adds the given counts to the table.
    pub fn load_transitions(&mut self, entries: Vec<(Vec<u8>, Vec<(u8, u32)>)>) {
        for (ctx, nonzero) in entries {
            let counts = self.patcher.transition_table.entry(ctx).or_default();
            for (b, c) in nonzero {
                match counts.iter_mut().find(|(byte, _)| *byte == b) {
                    Some((_, total)) => *total += c,
                    None => counts.push((b, c)),
                }
            }
        }
    }
}

/// GPT-2 byte-level printable form of raw bytes (the token strings of `get_vocab`).
#[pyfunction]
pub fn bytes_to_printable(data: &[u8]) -> String {
    byte_vocab().bytes_to_printable(data)
}

/// Inverse of `bytes_to_printable`; `ValueError` for a char outside the GPT-2 byte map.
#[pyfunction]
pub fn printable_to_bytes(text: &str) -> PyResult<Vec<u8>> {
    byte_vocab().printable_to_bytes(text).map_err(value_err)
}

fn byte_vocab() -> &'static Vocabulary {
    static VOCAB: std::sync::OnceLock<Vocabulary> = std::sync::OnceLock::new();
    VOCAB.get_or_init(Vocabulary::new_byte_level)
}

/// Rejects non-finite values, and negative ones unless `allow_negative` (the scorers would
/// otherwise clamp them to 0 silently).
fn check_param(name: &str, value: Option<f64>, allow_negative: bool) -> PyResult<()> {
    match value {
        Some(v) if !v.is_finite() || (v < 0.0 && !allow_negative) => Err(PyValueError::new_err(
            format!("{name} must be a finite{} number, got {v}", if allow_negative { "" } else { " non-negative" }),
        )),
        _ => Ok(()),
    }
}

/// Trains a byte-level BPE model. `vocab_size` counts every token, special tokens included; it
/// is an upper bound (training stops early when no eligible pair is left). `None` parameters take
/// the scorer's default; a parameter the scorer does not use raises `ValueError`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (corpus, vocab_size, scorer_type="frequency", min_frequency=None, alpha=None, beta=None, penalty=None, special_tokens=None))]
pub fn train_bpe(
    py: Python<'_>,
    corpus: Vec<String>,
    vocab_size: usize,
    scorer_type: &str,
    min_frequency: Option<u64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    penalty: Option<f64>,
    special_tokens: Option<Vec<String>>,
) -> PyResult<PyBpeModel> {
    check_param("alpha", alpha, false)?;
    check_param("beta", beta, true)?;
    check_param("penalty", penalty, false)?;
    let specials = special_tokens.unwrap_or_default();
    let merge_target = vocab_size.saturating_sub(specials.len());
    if merge_target < BASE_VOCAB_SIZE {
        return Err(PyValueError::new_err(format!(
            "vocab_size must be at least {BASE_VOCAB_SIZE} + the number of special tokens ({})",
            specials.len()
        )));
    }

    let scorer_norm = scorer_type.to_lowercase().replace('-', "_");
    // (default min_frequency, uses alpha, uses beta, uses penalty); defaults are the scorers' own.
    let (default_min, uses_alpha, uses_beta, uses_penalty) = match scorer_norm.as_str() {
        "frequency" | "bpe" | "wp_bpe_base" => (1, false, false, false),
        "wp_bpe_pig" => (WpBpePigScorer::default().min_frequency, false, false, false),
        "wordpiece" | "pmi" => (WordPieceScorer::default().min_frequency, true, false, false),
        "mdl" => (MdlScorer::default().min_frequency, false, false, true),
        "length_penalized" => (LengthPenalizedScorer::default().min_frequency, false, true, false),
        "wp_bpe_ft" => (WpBpeFtScorer::default().min_frequency, false, false, false),
        "wp_bpe_scp" => (WpBpeScpScorer::default().min_freq, true, false, false),
        "wp_bpe_npmi" => (WpBpeNpmiScorer::default().min_freq, true, false, false),
        "wp_bpe_ppig" => (WpBpePpigScorer::default().min_freq, false, false, true),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown scorer type: {scorer_type}"
            )));
        }
    };
    for (name, given, used) in [
        ("alpha", alpha.is_some(), uses_alpha),
        ("beta", beta.is_some(), uses_beta),
        ("penalty", penalty.is_some(), uses_penalty),
    ] {
        if given && !used {
            return Err(PyValueError::new_err(format!(
                "scorer {scorer_type:?} does not take {name}"
            )));
        }
    }
    let min_freq = min_frequency.unwrap_or(default_min).max(1);
    if scorer_norm == "wp_bpe_ft" && min_freq < 2 {
        return Err(PyValueError::new_err(
            "wp_bpe_ft is defined by min_frequency >= 2 (use wp_bpe_base for no cutoff)",
        ));
    }

    let model = py
        .allow_threads(|| -> Result<BottomUpModel, String> {
            let word_counts = extract_word_counts(&corpus).map_err(|e| e.to_string())?;
            let scorer: Box<dyn PairScorer> = match scorer_norm.as_str() {
                "frequency" | "bpe" => Box::new(BpeScorer::new()),
                "wordpiece" | "pmi" => Box::new(WordPieceScorer::new(
                    alpha.unwrap_or(WordPieceScorer::default().alpha),
                    min_freq,
                )),
                "mdl" => Box::new(MdlScorer::new(
                    penalty.unwrap_or(MdlScorer::default().entry_overhead_bits),
                    min_freq,
                )),
                "length_penalized" => Box::new(LengthPenalizedScorer::new(
                    beta.unwrap_or(LengthPenalizedScorer::default().beta),
                    min_freq,
                )),
                "wp_bpe_base" => Box::new(WpBpeBaseScorer::new()),
                "wp_bpe_ft" => Box::new(WpBpeFtScorer::new(min_freq)),
                "wp_bpe_pig" => Box::new(WpBpePigScorer::new(min_freq)),
                "wp_bpe_scp" => Box::new(WpBpeScpScorer::new(
                    min_freq,
                    alpha.unwrap_or(WpBpeScpScorer::default().discount),
                )),
                "wp_bpe_npmi" => Box::new(WpBpeNpmiScorer::new(
                    min_freq,
                    alpha.unwrap_or(WpBpeNpmiScorer::default().alpha),
                )),
                "wp_bpe_ppig" => Box::new(WpBpePpigScorer::new(
                    min_freq,
                    penalty.unwrap_or(WpBpePpigScorer::default().length_penalty),
                )),
                _ => unreachable!(),
            };
            let mut model = BottomUpTrainer::new(scorer.as_ref(), merge_target)
                .with_min_frequency(min_freq)
                .train(&word_counts)
                .map_err(|e| e.to_string())?;
            for special in &specials {
                model
                    .vocab
                    .add_special_token(special)
                    .map_err(|e| e.to_string())?;
            }
            Ok(model)
        })
        .map_err(PyValueError::new_err)?;

    Ok(PyBpeModel { model })
}

/// Trains a Unigram LM; `vocab_size` is an upper bound (the seed set may be smaller).
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
    m.add_function(wrap_pyfunction!(bytes_to_printable, m)?)?;
    m.add_function(wrap_pyfunction!(printable_to_bytes, m)?)?;
    Ok(())
}
