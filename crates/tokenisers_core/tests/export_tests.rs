use tempfile::tempdir;
use tokenisers_core::engine::bottom_up::BottomUpTrainer;
use tokenisers_core::engine::unigram::UnigramTrainer;
use tokenisers_core::export::HfTokenizerJson;
use tokenisers_core::scoring::BpeScorer;

#[test]
fn test_hf_bpe_export_json_schema() {
    let corpus = vec![
        (b"the quick brown fox jumps over the lazy dog".to_vec(), 50),
        (b"the lazy dog slept while the quick fox ran".to_vec(), 40),
    ];
    let scorer = BpeScorer::new();
    let model = BottomUpTrainer::new(&scorer, 270).train(&corpus).unwrap();

    let hf_json = HfTokenizerJson::from_bpe_model(&model, Some(0.1));
    let json_str = hf_json.to_json_string().expect("Failed to serialize to JSON");

    // Parse back with generic serde_json Value to verify structure
    let v: serde_json::Value = serde_json::from_str(&json_str).expect("Invalid JSON emitted");

    assert_eq!(v["version"], "1.0");
    assert_eq!(v["pre_tokenizer"]["type"], "ByteLevel");
    assert_eq!(v["pre_tokenizer"]["add_prefix_space"], false);
    assert_eq!(v["pre_tokenizer"]["trim_offsets"], false);

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
    assert_eq!(v["pre_tokenizer"]["type"], "ByteLevel");
    assert_eq!(v["pre_tokenizer"]["add_prefix_space"], false);
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

