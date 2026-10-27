use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use crate::engine::bottom_up::BottomUpModel;
use crate::engine::unigram::UnigramModel;
use crate::error::{Result, TokeniserError};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum PreTokenizerConfig {
    ByteLevel {
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum DecoderConfig {
    ByteLevel {
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum HfModelConfig {
    BPE {
        dropout: Option<f32>,
        unk_token: Option<String>,
        continuing_subword_prefix: Option<String>,
        end_of_word_suffix: Option<String>,
        fuse_unk: bool,
        byte_fallback: bool,
        ignore_merges: bool,
        vocab: HashMap<String, u32>,
        merges: Vec<String>,
    },
    Unigram {
        unk_id: Option<u32>,
        vocab: Vec<(String, f64)>,
        byte_fallback: bool,
    },
}

/// Canonical Hugging Face tokenizer.json representation with byte losslessness.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HfTokenizerJson {
    pub version: String,
    pub truncation: Option<serde_json::Value>,
    pub padding: Option<serde_json::Value>,
    pub added_tokens: Vec<AddedToken>,
    pub normalizer: Option<serde_json::Value>,
    pub pre_tokenizer: PreTokenizerConfig,
    pub post_processor: Option<serde_json::Value>,
    pub decoder: DecoderConfig,
    pub model: HfModelConfig,
}

impl HfTokenizerJson {
    /// Creates a canonical BPE tokenizer.json from a BottomUpModel.
    pub fn from_bpe_model(model: &BottomUpModel, dropout: Option<f32>) -> Self {
        let vocab_map = model.vocab.to_string_map();
        let mut merges = Vec::with_capacity(model.merges.len());

        for &(u, v, _) in &model.merges {
            let bytes_u = model.vocab.get_bytes(u).unwrap_or(&[]);
            let bytes_v = model.vocab.get_bytes(v).unwrap_or(&[]);
            let str_u = model.vocab.bytes_to_printable(bytes_u);
            let str_v = model.vocab.bytes_to_printable(bytes_v);
            merges.push(format!("{str_u} {str_v}"));
        }

        let mut string_vocab: HashMap<String, u32> = HashMap::with_capacity(vocab_map.len());
        for (k, v) in vocab_map {
            string_vocab.insert(k, v);
        }

        let added_tokens = model
            .vocab
            .special_tokens()
            .iter()
            .map(|st| AddedToken {
                id: st.id,
                content: st.content.clone(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: true,
            })
            .collect();

        Self {
            version: "1.0".into(),
            truncation: None,
            padding: None,
            added_tokens,
            normalizer: None,
            pre_tokenizer: PreTokenizerConfig::ByteLevel {
                add_prefix_space: false,
                trim_offsets: false,
                use_regex: true,
            },
            post_processor: None,
            decoder: DecoderConfig::ByteLevel {
                add_prefix_space: false,
                trim_offsets: false,
                use_regex: true,
            },
            model: HfModelConfig::BPE {
                dropout,
                unk_token: None,
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                fuse_unk: false,
                byte_fallback: false,
                ignore_merges: false,
                vocab: string_vocab,
                merges,
            },
        }
    }

    /// Creates a canonical Unigram tokenizer.json from a UnigramModel.
    pub fn from_unigram_model(model: &UnigramModel, unk_id: Option<u32>) -> Self {
        let mut unigram_vocab = Vec::with_capacity(model.vocab.len());
        for (id, bytes) in model.vocab.id_to_bytes_map().iter().enumerate() {
            let piece = model.vocab.bytes_to_printable(bytes);
            let score = if id < model.log_probs.len() {
                model.log_probs[id]
            } else {
                -15.0
            };
            unigram_vocab.push((piece, score));
        }

        let added_tokens = model
            .vocab
            .special_tokens()
            .iter()
            .map(|st| AddedToken {
                id: st.id,
                content: st.content.clone(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: true,
            })
            .collect();

        Self {
            version: "1.0".into(),
            truncation: None,
            padding: None,
            added_tokens,
            normalizer: None,
            pre_tokenizer: PreTokenizerConfig::ByteLevel {
                add_prefix_space: false,
                trim_offsets: false,
                use_regex: true,
            },
            post_processor: None,
            decoder: DecoderConfig::ByteLevel {
                add_prefix_space: false,
                trim_offsets: false,
                use_regex: true,
            },
            model: HfModelConfig::Unigram {
                unk_id,
                vocab: unigram_vocab,
                byte_fallback: false,
            },
        }
    }

    /// Exports the tokenizer configuration to a file or directory.
    pub fn export_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let target_file = if path.is_dir() || path.extension().is_none() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };

        if let Some(parent) = target_file.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(target_file)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Serializes tokenizer config to formatted JSON string.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(TokeniserError::Json)
    }
}
