use std::collections::BTreeMap;

use tokenisers_core::curricula::super_bpe::SuperBpeTrainer;
use tokenisers_core::engine::bottom_up::{BottomUpModel, BottomUpTrainer};
use tokenisers_core::scoring::BpeScorer;
use tokenisers_core::vocab::{bytes_char, Vocabulary, BASE_VOCAB_SIZE};

#[test]
fn test_base_vocab_size_is_256() {
    let vocab = Vocabulary::new_byte_level();
    assert_eq!(vocab.len(), BASE_VOCAB_SIZE);
    for b in 0..=255u8 {
        assert_eq!(vocab.get_bytes(b as u32), Some(&[b][..]));
        assert_eq!(vocab.get_id(&[b]), Some(b as u32));
    }
}

#[test]
fn test_bytes_char_bijection() {
    let (byte_to_char, char_to_byte) = bytes_char();
    let mut seen_chars = std::collections::HashSet::new();

    for b in 0..=255u8 {
        let ch = byte_to_char[b as usize];
        assert_ne!(ch, '\0', "Byte {b} mapped to null character");
        assert!(seen_chars.insert(ch), "Duplicate character mapping for byte {b}");

        let cp = ch as usize;
        assert!(cp < 512, "Code point {cp} exceeds lookup table bound");
        assert_eq!(char_to_byte[cp], Some(b), "Inverse mapping failed for byte {b}");
    }
}

#[test]
fn test_printable_conversion_roundtrip() {
    let vocab = Vocabulary::new_byte_level();
    let all_bytes: Vec<u8> = (0..=255u8).collect();
    let printable = vocab.bytes_to_printable(&all_bytes);
    let recovered = vocab.printable_to_bytes(&printable).expect("Printable decoding failed");
    assert_eq!(all_bytes, recovered);
}

const ASCII_SAMPLES: [&str; 4] = [
    "Hello, World!",
    "def main():\n    return 42\n",
    "1234567890!@#$%^&*()_+-=[]{}|;':\",./<>?",
    "   leading and trailing spaces   \t\r\n",
];

const MULTILINGUAL_SAMPLES: [&str; 8] = [
    "你好，世界！这是一个多语言分词器测试。",
    "こんにちは世界！トークナイザーのテストです。",
    "Привет, мир! Тестирование токенизатора.",
    "مرحبا بالعالم! اختبار مجزئ الكلمات.",
    "नमस्ते दुनिया! टोकनाइज़र परीक्षण।",
    "Γειά σου κόσμε! Δοκιμή κατακερματιστή.",
    "한국어 토크나이저 테스트 문장입니다.",
    "Héllò Wörld! Übergrößenträger für Café & Naïve.",
];

const EMOJI_SAMPLES: [&str; 3] = [
    "😀 😃 😄 😁 😆 😅 😂 🤣",
    "👍🏽 👨‍👩‍👧‍👦 🏳️‍🌈 🧑🏽‍💻",
    "Complexity: O(1) 🚀 Speed: 100x ⚡ Quality: 💯",
];

/// BPE model trained (via the same segmentation as the Python bindings) on every sample, so
/// round trips go through learned multi-byte merges rather than the identity byte map.
fn trained_model() -> BottomUpModel {
    let mut counts: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    for text in ASCII_SAMPLES.iter().chain(&MULTILINGUAL_SAMPLES).chain(&EMOJI_SAMPLES) {
        for seg in SuperBpeTrainer::segment_text(text) {
            *counts.entry(seg).or_insert(0) += 3;
        }
    }
    let corpus: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
    BottomUpTrainer::new(&BpeScorer, 700).train(&corpus).unwrap()
}

fn assert_roundtrip_uses_merges(model: &BottomUpModel, text: &str) {
    let tokens = model.encode(text.as_bytes());
    assert!(tokens.len() < text.len(), "no learned merge applied to {text:?}");
    assert!(tokens.iter().any(|&t| t as usize >= BASE_VOCAB_SIZE));
    assert_eq!(model.decode_lossless(&tokens).expect("UTF-8 decoding failed"), text);
}

#[test]
fn test_ascii_roundtrip_losslessness() {
    let model = trained_model();
    for s in ASCII_SAMPLES {
        assert_roundtrip_uses_merges(&model, s);
    }
}

#[test]
fn test_multilingual_utf8_losslessness() {
    let model = trained_model();
    for text in MULTILINGUAL_SAMPLES {
        assert_roundtrip_uses_merges(&model, text);
    }
}

#[test]
fn test_emoji_and_zwj_sequences_losslessness() {
    let model = trained_model();
    for text in EMOJI_SAMPLES {
        assert_roundtrip_uses_merges(&model, text);
    }
}

#[test]
fn test_unseen_text_losslessness() {
    let model = trained_model();
    for text in ["Hello Wörld, 世界 🚀 return 42;", "\u{0}\u{7f}\r\n\t  zzz", ""] {
        let tokens = model.encode(text.as_bytes());
        assert_eq!(model.decode_lossless(&tokens).unwrap(), text);
    }
}

#[test]
fn test_arbitrary_binary_and_control_bytes_losslessness() {
    let vocab = Vocabulary::new_byte_level();
    // Non-UTF8 arbitrary byte stream containing null bytes and high bytes
    let raw_binary = vec![0x00, 0xFF, 0x80, 0xFE, 0x01, 0x7F, 0xC0, 0xAF, 0x1B, 0x0A];
    let tokens = vocab.encode_bytes_fallback(&raw_binary);
    let decoded_bytes = vocab.decode(&tokens).unwrap();
    assert_eq!(raw_binary, decoded_bytes);

    let model = trained_model();
    let mut mixed = "Hello, World! 你好".as_bytes().to_vec();
    mixed.extend_from_slice(&raw_binary);
    mixed.extend_from_slice("Привет".as_bytes());
    let tokens = model.encode(&mixed);
    assert!(tokens.len() < mixed.len());
    assert_eq!(model.decode(&tokens).unwrap(), mixed);
}

#[test]
fn test_special_tokens_handling() {
    let mut vocab = Vocabulary::new_byte_level();
    let id_eos = vocab.add_special_token("<|endoftext|>").expect("Failed to add special token");
    let id_pad = vocab.add_special_token("<pad>").expect("Failed to add special token");

    assert!(vocab.is_special_token(id_eos));
    assert!(vocab.is_special_token(id_pad));
    assert!(!vocab.is_special_token(0));

    // Duplicate special token should error
    let err = vocab.add_special_token("<|endoftext|>");
    assert!(err.is_err());
}
