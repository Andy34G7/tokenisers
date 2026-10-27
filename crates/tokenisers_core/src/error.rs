use thiserror::Error;

pub type Result<T> = std::result::Result<T, TokeniserError>;

#[derive(Error, Debug)]
pub enum TokeniserError {
    #[error("Target vocabulary size {0} is smaller than base alphabet size 256")]
    VocabSizeTooSmall(usize),

    #[error("Special token '{0}' already exists in vocabulary")]
    DuplicateSpecialToken(String),

    #[error("Corpus is empty: cannot train tokenizer on zero bytes")]
    EmptyCorpus,

    #[error("Node index {0} is out of bounds for arena of size {1}")]
    ArenaIndexOutOfBounds(u32, usize),

    #[error("Corrupted arena link: node {0} next points to {1}, but {1} prev points to {2}")]
    CorruptedArenaLink(u32, u32, u32),

    #[error("Token ID {0} not found in vocabulary")]
    InvalidTokenId(u32),

    #[error("Invalid byte representation or character mapping: {0}")]
    InvalidByteMapping(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("UTF-8 decoding error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("UTF-8 str error: {0}")]
    Utf8Str(#[from] std::str::Utf8Error),
}
