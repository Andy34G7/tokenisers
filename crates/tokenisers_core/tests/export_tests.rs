use tempfile::tempdir;
use tokenisers_core::engine::bottom_up::BottomUpTrainer;
use tokenisers_core::engine::unigram::UnigramTrainer;
use tokenisers_core::export::HfTokenizerJson;
use tokenisers_core::scoring::BpeScorer;

/// Native encoding has no GPT-2 regex: HF must split exactly `segment_text` segments (isolated
/// whitespace runs) and then byte-map them (`use_regex: false`). Parity is checked against the
/// real HF tokenizer in hf_parity_tests.rs.
fn assert_segment_then_byte_level(pre: &serde_json::Value) {
    assert_eq!(pre["type"], "Sequence");
    let steps = pre["pretokenizers"].as_array().unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["type"], "Split");
    assert_eq!(steps[0]["behavior"], "Isolated");
    assert!(steps[0]["pattern"]["Regex"].as_str().unwrap().contains("\\x{3000}"));
    assert_eq!(steps[1]["type"], "ByteLevel");
    assert_eq!(steps[1]["add_prefix_space"], false);
    assert_eq!(steps[1]["use_regex"], false);
}

#[test]
fn test_hf_bpe_export_json_schema() {
    // Word counts as the Python binding builds them: one `segment_text` segment per word.
    let corpus: Vec<(Vec<u8>, u64)> = ["the", " ", "quick", "brown", "fox", "lazy", "dog"]
        .iter()
        .map(|w| (w.as_bytes().to_vec(), 50))
        .collect();
    let scorer = BpeScorer::new();
    let model = BottomUpTrainer::new(&scorer, 270).train(&corpus).unwrap();
    assert!(model.pretokenize);

    let hf_json = HfTokenizerJson::from_bpe_model(&model, Some(0.1));
    let json_str = hf_json.to_json_string().expect("Failed to serialize to JSON");

    // Parse back with generic serde_json Value to verify structure
    let v: serde_json::Value = serde_json::from_str(&json_str).expect("Invalid JSON emitted");

    assert_eq!(v["version"], "1.0");
    assert_segment_then_byte_level(&v["pre_tokenizer"]);

    assert_eq!(v["decoder"]["type"], "ByteLevel");
    assert_eq!(v["decoder"]["add_prefix_space"], false);

    assert_eq!(v["model"]["type"], "BPE");
    assert_eq!(v["model"]["dropout"], 0.1);
    assert!(v["model"]["vocab"].is_object());
    assert!(v["model"]["merges"].is_array());

    let merges = v["model"]["merges"].as_array().unwrap();
    assert_eq!(merges.len(), model.merges.len());

    // Verify merge string format: "token_u token_v"
    if let Some(first_merge) = merges.first() {
        let s = first_merge.as_str().unwrap();
        assert!(s.contains(' '), "Merge entry '{s}' must have whitespace separator");
    }
}

#[test]
fn test_hf_unigram_export_json_schema() {
    let corpus = vec![
        (b"machine learning and natural language processing".to_vec(), 30),
        (b"deep neural networks for language models".to_vec(), 25),
    ];
    let trainer = UnigramTrainer::new(270, 0.2);
    let model = trainer.train(&corpus).unwrap();

    let hf_json = HfTokenizerJson::from_unigram_model(&model, None);
    let json_str = hf_json.to_json_string().expect("Failed to serialize to JSON");

    let v: serde_json::Value = serde_json::from_str(&json_str).expect("Invalid JSON emitted");

    assert_eq!(v["version"], "1.0");
    assert_segment_then_byte_level(&v["pre_tokenizer"]);
    assert_eq!(v["decoder"]["type"], "ByteLevel");

    assert_eq!(v["model"]["type"], "Unigram");
    assert!(v["model"]["vocab"].is_array());
    let vocab_arr = v["model"]["vocab"].as_array().unwrap();
    assert!(vocab_arr.len() >= 256);

    // Each vocab item is [piece, score]
    for item in vocab_arr.iter().take(10) {
        let entry = item.as_array().expect("Vocab entry must be a 2-element array");
        assert_eq!(entry.len(), 2);
        assert!(entry[0].is_string());
        assert!(entry[1].is_number());
    }
}

#[test]
fn test_export_to_file_and_directory() {
    let corpus = vec![
        (b"hello world test serialization".to_vec(), 10),
    ];
    let scorer = BpeScorer::new();
    let model = BottomUpTrainer::new(&scorer, 260).train(&corpus).unwrap();
    let hf_json = HfTokenizerJson::from_bpe_model(&model, None);

    let temp_dir = tempdir().expect("Failed to create tempdir");

    // 1. Export to directory (creates tokenizer.json inside directory)
    hf_json.export_to_file(temp_dir.path()).expect("Directory export failed");
    let file_path = temp_dir.path().join("tokenizer.json");
    assert!(file_path.exists());

    let content = std::fs::read_to_string(&file_path).expect("Read exported file failed");
    let parsed: HfTokenizerJson = serde_json::from_str(&content).expect("Failed to parse back HfTokenizerJson");
    assert_eq!(parsed.version, "1.0");

    // 2. Export to explicit custom file path
    let custom_file = temp_dir.path().join("custom_model.json");
    hf_json.export_to_file(&custom_file).expect("File export failed");
    assert!(custom_file.exists());
}

#[test]
fn test_hf_export_populates_special_tokens() {
    let corpus = vec![
        (b"special token handling test".to_vec(), 10),
    ];
    let scorer = BpeScorer::new();
    let mut model = BottomUpTrainer::new(&scorer, 260).train(&corpus).unwrap();
    model.vocab.add_special_token("<|endoftext|>").unwrap();
    model.vocab.add_special_token("<unk>").unwrap();

    let hf_json = HfTokenizerJson::from_bpe_model(&model, None);
    let json_str = hf_json.to_json_string().expect("Failed to serialize to JSON");
    let v: serde_json::Value = serde_json::from_str(&json_str).expect("Invalid JSON emitted");

    let added_tokens = v["added_tokens"].as_array().expect("added_tokens must be an array");
    assert_eq!(added_tokens.len(), 2);
    let contents: Vec<&str> = added_tokens.iter().map(|t| t["content"].as_str().unwrap()).collect();
    assert!(contents.contains(&"<|endoftext|>"));
    assert!(contents.contains(&"<unk>"));
    for t in added_tokens {
        assert_eq!(t["special"], true);
    }
}


#[test]
fn super_bpe_export_encodes_whole_text() {
    let lines = vec!["the quick brown fox jumps over the lazy dog"; 4];
    let config = tokenisers_core::curricula::SuperBpeConfig::new(270, 285).unwrap();
    let scorer = BpeScorer::new();
    let model = tokenisers_core::curricula::SuperBpeTrainer::new(config, &scorer)
        .train(&lines)
        .unwrap();
    let exported = HfTokenizerJson::from_bpe_model(&model, None).to_json_string().unwrap();
    let v: serde_json::Value = serde_json::from_str(&exported).unwrap();
    assert_eq!(v["pre_tokenizer"]["type"], "ByteLevel");
    assert_eq!(v["pre_tokenizer"]["use_regex"], false);

    // A loader recovers the encoding mode from the file.
    let reparsed: HfTokenizerJson = serde_json::from_str(&exported).unwrap();
    assert!(!reparsed.pretokenize());
    let plain = BottomUpTrainer::new(&scorer, 270).train(&[(b"ab".to_vec(), 3)]).unwrap();
    let plain_json = HfTokenizerJson::from_bpe_model(&plain, None).to_json_string().unwrap();
    assert!(serde_json::from_str::<HfTokenizerJson>(&plain_json).unwrap().pretokenize());
}

#[test]
fn bpe_trained_on_multi_segment_words_exports_whole_text() {
    // Training words that span whitespace can teach merges like ("b", " "); pre-splitting would
    // make them unreachable, so such a model encodes and exports whole text.
    let scorer = BpeScorer::new();
    let model = BottomUpTrainer::new(&scorer, 258).train(&[(b"ab ab".to_vec(), 3)]).unwrap();
    assert!(!model.pretokenize);
    let json = HfTokenizerJson::from_bpe_model(&model, None).to_json_string().unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["pre_tokenizer"]["type"], "ByteLevel");
    assert_eq!(v["pre_tokenizer"]["use_regex"], false);
}

#[test]
fn bpe_export_is_byte_identical_across_calls() {
    // The vocab is serialised in a fixed (sorted) order, not hash-map order.
    let corpus = vec![(b"deterministic exports please".to_vec(), 7)];
    let scorer = BpeScorer::new();
    let model = BottomUpTrainer::new(&scorer, 270).train(&corpus).unwrap();
    let a = HfTokenizerJson::from_bpe_model(&model, None).to_json_string().unwrap();
    let b = HfTokenizerJson::from_bpe_model(&model, None).to_json_string().unwrap();
    assert_eq!(a, b);
}
