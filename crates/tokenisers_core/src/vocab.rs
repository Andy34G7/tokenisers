use ahash::{AHashMap, AHashSet};
use crate::error::{Result, TokeniserError};

pub type TokenId = u32;

pub const BASE_VOCAB_SIZE: usize = 256;

/// Generates the canonical GPT-2 bijective byte-to-char mapping.
///
/// 188 bytes map directly to their Unicode code points:
/// - Printable ASCII: 33..=126 ('!'..='~')
/// - Latin-1 Supplement: 161..=172 ('¡'..='¬') and 174..=255 ('®'..='ÿ')
///
/// The remaining 68 non-printable, control, and whitespace bytes (0..=32, 127..=160, 173)
/// map bijectively to Unicode code points 256..=323.
pub fn bytes_char() -> ([char; 256], [Option<u8>; 512]) {
    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = [None; 512];

    let mut bs: Vec<u8> = Vec::with_capacity(256);

    // Printable ASCII
    for b in b'!'..=b'~' {
        bs.push(b);
    }
    // Latin-1 Supplement printable
    for b in 0xA1..=0xAC {
        bs.push(b);
    }
    for b in 0xAE..=0xFF {
        bs.push(b);
    }

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    for (&b, &c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(c).expect("Valid Unicode character");
        byte_to_char[b as usize] = ch;
        if (c as usize) < 512 {
            char_to_byte[c as usize] = Some(b);
        }
    }

    (byte_to_char, char_to_byte)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecialToken {
    pub id: TokenId,
    pub content: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Vocabulary {
    id_to_bytes: Vec<Vec<u8>>,
    bytes_to_id: AHashMap<Vec<u8>, TokenId>,
    special_tokens: Vec<SpecialToken>,
    special_token_ids: AHashSet<TokenId>,
    byte_to_char: [char; 256],
    char_to_byte: [Option<u8>; 512],
}

pub type Vocab = Vocabulary;

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new_byte_level()
    }
}

impl Vocabulary {
    /// Initializes a base byte-level vocabulary with 256 single-byte tokens (0..=255).
    pub fn new_byte_level() -> Self {
        let (byte_to_char, char_to_byte) = bytes_char();
        let mut id_to_bytes = Vec::with_capacity(BASE_VOCAB_SIZE);
        let mut bytes_to_id = AHashMap::with_capacity(BASE_VOCAB_SIZE);

        for b in 0..=255u8 {
            let bytes = vec![b];
            bytes_to_id.insert(bytes.clone(), b as TokenId);
            id_to_bytes.push(bytes);
        }

        Self {
            id_to_bytes,
            bytes_to_id,
            special_tokens: Vec::new(),
            special_token_ids: AHashSet::new(),
            byte_to_char,
            char_to_byte,
        }
    }

    /// Alias for new_byte_level.
    pub fn new() -> Self {
        Self::new_byte_level()
    }

    /// Returns the total number of tokens in the vocabulary (including base bytes and special tokens).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.id_to_bytes.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.id_to_bytes.is_empty()
    }

    /// Adds a merged or compound token to the vocabulary.
    /// If the token already exists, returns its existing TokenId.
    pub fn add_token(&mut self, bytes: Vec<u8>) -> TokenId {
        if let Some(&id) = self.bytes_to_id.get(&bytes) {
            return id;
        }

        let new_id = self.id_to_bytes.len() as TokenId;
        self.bytes_to_id.insert(bytes.clone(), new_id);
        self.id_to_bytes.push(bytes);
        new_id
    }

    /// Adds a special token (e.g. `<|endoftext|>`, `<pad>`) to the vocabulary.
    ///
    /// A special token is known by its raw content, every other token by the GPT-2 printable form
    /// of its bytes (`get_vocab`, tokenizer.json keys). Content that is the printable form of a
    /// different existing token (e.g. "é", the key of byte 0xE9) would make those keys collide, so
    /// it is rejected.
    pub fn add_special_token(&mut self, content: &str) -> Result<TokenId> {
        let bytes = content.as_bytes().to_vec();
        if self.special_tokens.iter().any(|s| s.content == content) {
            return Err(TokeniserError::DuplicateSpecialToken(content.to_string()));
        }
        if let Ok(key_bytes) = self.printable_to_bytes(content) {
            if key_bytes != bytes {
                if let Some(other) = self.get_id(&key_bytes) {
                    return Err(TokeniserError::InvalidConfiguration(format!(
                        "special token {content:?} collides with the byte-level token string of \
                         id {other} (bytes {key_bytes:02X?})"
                    )));
                }
            }
        }

        let id = if let Some(&existing_id) = self.bytes_to_id.get(&bytes) {
            self.special_token_ids.insert(existing_id);
            existing_id
        } else {
            let new_id = self.id_to_bytes.len() as TokenId;
            self.bytes_to_id.insert(bytes.clone(), new_id);
            self.id_to_bytes.push(bytes.clone());
            self.special_token_ids.insert(new_id);
            new_id
        };

        self.special_tokens.push(SpecialToken {
            id,
            content: content.to_string(),
            bytes,
        });

        Ok(id)
    }

    /// Looks up a TokenId by its exact byte sequence.
    #[inline(always)]
    pub fn get_id(&self, bytes: &[u8]) -> Option<TokenId> {
        self.bytes_to_id.get(bytes).copied()
    }

    /// Looks up the byte slice corresponding to a TokenId.
    #[inline(always)]
    pub fn get_bytes(&self, id: TokenId) -> Option<&[u8]> {
        self.id_to_bytes.get(id as usize).map(|v| v.as_slice())
    }

    /// Checks if a TokenId is a registered special token.
    #[inline(always)]
    pub fn is_special_token(&self, id: TokenId) -> bool {
        self.special_token_ids.contains(&id)
    }

    /// Returns a reference to all special tokens.
    #[inline(always)]
    pub fn special_tokens(&self) -> &[SpecialToken] {
        &self.special_tokens
    }

    /// Converts raw bytes into the canonical GPT-2 printable Unicode string.
    pub fn bytes_to_printable(&self, bytes: &[u8]) -> String {
        bytes.iter().map(|&b| self.byte_to_char[b as usize]).collect()
    }

    /// Decodes a canonical GPT-2 printable Unicode string back into original raw bytes.
    pub fn printable_to_bytes(&self, s: &str) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(s.len());
        for c in s.chars() {
            let cp = c as usize;
            if cp < 512 {
                if let Some(b) = self.char_to_byte[cp] {
                    bytes.push(b);
                    continue;
                }
            }
            return Err(TokeniserError::InvalidByteMapping(format!(
                "Character '{c}' (U+{:04X}) not in GPT-2 byte mapping",
                c as u32
            )));
        }
        Ok(bytes)
    }

    /// Concatenates token byte slices into raw bytes.
    /// Returns `InvalidTokenId` for any id outside the vocabulary.
    pub fn decode(&self, ids: &[TokenId]) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            let bytes = self.get_bytes(id).ok_or(TokeniserError::InvalidTokenId(id))?;
            out.extend_from_slice(bytes);
        }
        Ok(out)
    }

    /// Decodes a sequence of TokenIds losslessly to a UTF-8 String.
    /// Returns `InvalidTokenId` for unknown ids and `Utf8` if the bytes are not valid UTF-8.
    pub fn decode_lossless(&self, ids: &[TokenId]) -> Result<String> {
        let bytes = self.decode(ids)?;
        String::from_utf8(bytes).map_err(TokeniserError::Utf8)
    }

    /// Fallback encoding: maps each raw byte directly to its single-byte TokenId (0..=255).
    pub fn encode_bytes_fallback(&self, bytes: &[u8]) -> Vec<TokenId> {
        bytes.iter().map(|&b| b as TokenId).collect()
    }

    /// Returns the raw mapping of TokenId to bytes.
    pub fn id_to_bytes_map(&self) -> &[Vec<u8>] {
        &self.id_to_bytes
    }

    /// Returns a map from GPT-2 printable string to TokenId.
    pub fn to_string_map(&self) -> AHashMap<String, TokenId> {
        let mut map = AHashMap::with_capacity(self.len());
        for (id, bytes) in self.id_to_bytes.iter().enumerate() {
            let s = self.bytes_to_printable(bytes);
            map.insert(s, id as TokenId);
        }
        map
    }
}
