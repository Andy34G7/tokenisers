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

#[test]
fn test_ascii_roundtrip_losslessness() {
    let vocab = Vocabulary::new_byte_level();
    let test_strings = [
        "Hello, World!",
        "def main():\n    return 42\n",
        "1234567890!@#$%^&*()_+-=[]{}|;':\",./<>?",
        "   leading and trailing spaces   \t\r\n",
    ];

    for s in test_strings {
        let bytes = s.as_bytes();
        let tokens = vocab.encode_bytes_fallback(bytes);
        assert_eq!(tokens.len(), bytes.len());
        let decoded = vocab.decode_lossless(&tokens).expect("UTF-8 decoding failed");
        assert_eq!(s, decoded);
    }
}

#[test]
fn test_multilingual_utf8_losslessness() {
    let vocab = Vocabulary::new_byte_level();
    let multilingual_samples = [
        "你好，世界！这是一个多语言分词器测试。",
        "こんにちは世界！トークナイザーのテストです。",
        "Привет, мир! Тестирование токенизатора.",
        "مرحبا بالعالم! اختبار مجزئ الكلمات.",
        "नमस्ते दुनिया! टोकनाइज़र परीक्षण।",
        "Γειά σου κόσμε! Δοκιμή κατακερματιστή.",
        "한국어 토크나이저 테스트 문장입니다.",
        "Héllò Wörld! Übergrößenträger für Café & Naïve.",
    ];

    for text in multilingual_samples {
        let bytes = text.as_bytes();
        let tokens = vocab.encode_bytes_fallback(bytes);
        let decoded = vocab.decode_lossless(&tokens).expect("UTF-8 decoding failed");
        assert_eq!(text, decoded);
    }
}

#[test]
fn test_emoji_and_zwj_sequences_losslessness() {
    let vocab = Vocabulary::new_byte_level();
    let emoji_samples = [
        "😀 😃 😄 😁 😆 😅 😂 🤣",
        "👍🏽 👨‍👩‍👧‍👦 🏳️‍🌈 🧑🏽‍💻",
        "Complexity: O(1) 🚀 Speed: 100x ⚡ Quality: 💯",
    ];

    for text in emoji_samples {
        let bytes = text.as_bytes();
        let tokens = vocab.encode_bytes_fallback(bytes);
        let decoded = vocab.decode_lossless(&tokens).expect("UTF-8 decoding failed");
        assert_eq!(text, decoded);
    }
}

#[test]
fn test_arbitrary_binary_and_control_bytes_losslessness() {
    let vocab = Vocabulary::new_byte_level();
    // Non-UTF8 arbitrary byte stream containing null bytes and high bytes
    let raw_binary = vec![0x00, 0xFF, 0x80, 0xFE, 0x01, 0x7F, 0xC0, 0xAF, 0x1B, 0x0A];
    let tokens = vocab.encode_bytes_fallback(&raw_binary);
    let decoded_bytes = vocab.decode(&tokens);
    assert_eq!(raw_binary, decoded_bytes);
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
