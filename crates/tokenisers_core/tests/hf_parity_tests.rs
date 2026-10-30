//! HF parity: an exported tokenizer.json loaded by the real Hugging Face `tokenizers` crate must
//! produce exactly the ids native encoding produces, for every exportable model family.

use std::collections::BTreeMap;

use tokenisers_core::curricula::super_bpe::{SuperBpeConfig, SuperBpeTrainer};
use tokenisers_core::engine::bottom_up::{BottomUpModel, BottomUpTrainer};
use tokenisers_core::engine::unigram::UnigramTrainer;
use tokenisers_core::export::huggingface::{HfTokenizerJson, WHITESPACE_RANGES};
use tokenisers_core::scoring::*;
use tokenisers_core::vocab::TokenId;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer};

const DOCS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog. Don't stop!",
    "hello, world! hello, world! hello,world -- (parens) [brackets] {braces}",
    "def foo(x):\n    return x + 1  # comment\n\n\tif x: pass",
    "price: $123.45, abc123 xyz789 1234567 3.14159e-10",
    "Héllo wörld, naïve café résumé façade",
    "日本語のテキスト、中文文本。한국어 텍스트",
    "中文　文本、中文。日本語　の　テキスト",
    "emoji 🚀🔥💯 party 🎉🎉 👩‍👩‍👧",
    "   leading spaces   and   runs\t\ttabs\n\nnewlines\r\n",
    "non\u{a0}breaking\u{a0}space and\u{2003}em\u{2009}thin\u{3000}ideographic",
    "it's they're we've I'm y'all",
];

const TESTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "hello, world!",
    "hello,world",
    "def foo(x):\n    return x + 1  # comment",
    "fn main() { println!(\"{}\", a[0] + b.c); }",
    "abc123 xyz789",
    "Héllo wörld, naïve café",
    "日本語のテキスト 中文文本",
    "中文　文本、中文。",
    "中　の、",
    "emoji 🚀🔥💯 party 🎉 👩‍👩‍👧",
    "a  b   c    d",
    "   leading",
    "trailing   ",
    "it's they're",
    "tabs\t\tand\n\nnewlines\r\n",
    "non\u{a0}breaking\u{3000}ideo\u{2028}line\u{85}nel\u{1680}ogham\u{205f}math",
    "zero\u{200b}width\u{feff}bom\u{180e}mvs\u{1c}fs",
    "",
    " ",
    "x",
];

fn corpus_docs() -> Vec<String> {
    DOCS.iter()
        .enumerate()
        .flat_map(|(i, d)| (0..(5 + i)).map(move |r| format!("{d} {r}")))
        .collect()
}

/// Word counts exactly as the Python binding builds them (segment_text segments).
fn word_counts(docs: &[String]) -> Vec<(Vec<u8>, u64)> {
    let mut counts: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    for d in docs {
        for seg in SuperBpeTrainer::segment_text(d) {
            *counts.entry(seg).or_insert(0) += 1;
        }
    }
    counts.into_iter().collect()
}

fn hf(json: &HfTokenizerJson) -> Tokenizer {
    json.to_json_string()
        .unwrap()
        .parse()
        .expect("HF tokenizers must load the exported tokenizer.json")
}

fn hf_ids(tok: &Tokenizer, s: &str) -> Vec<TokenId> {
    tok.encode(s, false).unwrap().get_ids().to_vec()
}

fn assert_parity(name: &str, tok: &Tokenizer, native: impl Fn(&str) -> Vec<TokenId>) {
    for s in TESTS {
        assert_eq!(native(s), hf_ids(tok, s), "[{name}] HF ids differ from native for {s:?}");
    }
}

fn bpe_models() -> Vec<(&'static str, BottomUpModel)> {
    let wc = word_counts(&corpus_docs());
    let train = |scorer: &dyn PairScorer| BottomUpTrainer::new(scorer, 420).train(&wc).unwrap();
    vec![
        ("bpe_frequency", train(&BpeScorer)),
        ("wordpiece", train(&WordPieceScorer::default())),
        ("wp_bpe_base", train(&WpBpeBaseScorer::default())),
        ("wp_bpe_ft", train(&WpBpeFtScorer::default())),
        ("wp_bpe_pig", train(&WpBpePigScorer::default())),
        ("wp_bpe_scp", train(&WpBpeScpScorer::default())),
        ("wp_bpe_npmi", train(&WpBpeNpmiScorer::default())),
        ("wp_bpe_ppig", train(&WpBpePpigScorer::default())),
        ("mdl", train(&MdlScorer::default())),
        ("length_penalized", train(&LengthPenalizedScorer::default())),
    ]
}

#[test]
fn bpe_family_export_matches_native_ids() {
    for (name, model) in bpe_models() {
        assert!(model.merges.len() > 20, "[{name}] trained too few merges to be a real test");
        let tok = hf(&HfTokenizerJson::from_bpe_model(&model, None));
        assert_parity(name, &tok, |s| model.encode(s.as_bytes()));
    }
}

#[test]
fn super_bpe_export_matches_native_ids() {
    let docs = corpus_docs();
    let lines: Vec<&str> = docs.iter().map(String::as_str).collect();
    let config = SuperBpeConfig::new(330, 460).unwrap();
    let model = SuperBpeTrainer::new(config, &BpeScorer).train(&lines).unwrap();
    assert!(!model.pretokenize, "stage-2 SuperBPE model must encode whole text");
    let crossing = model.vocab.id_to_bytes_map().iter().any(|t| {
        let s = String::from_utf8_lossy(t);
        s.chars().any(char::is_whitespace) && s.chars().any(|c| !c.is_whitespace())
    });
    assert!(crossing, "test corpus must produce whitespace-crossing SuperBPE tokens");
    let tok = hf(&HfTokenizerJson::from_bpe_model(&model, None));
    assert_parity("super_bpe", &tok, |s| model.encode(s.as_bytes()));
}

#[test]
fn unigram_export_matches_native_ids() {
    let model = UnigramTrainer::new(500, 0.2).train(&word_counts(&corpus_docs())).unwrap();
    let tok = hf(&HfTokenizerJson::from_unigram_model(&model, None));
    assert_parity("unigram", &tok, |s| model.encode(s.as_bytes()));
}

/// Reference for "split on special tokens (leftmost-longest, like HF's added vocabulary), then
/// encode each remaining piece".
fn split_specials_then(model: &BottomUpModel, s: &str) -> Vec<TokenId> {
    let specials = model.vocab.special_tokens();
    let (mut out, mut piece_start, mut i) = (Vec::new(), 0, 0);
    while i < s.len() {
        let hit = specials
            .iter()
            .filter(|st| s[i..].starts_with(st.content.as_str()))
            .max_by_key(|st| st.content.len());
        match hit {
            Some(st) => {
                out.extend(model.encode(&s.as_bytes()[piece_start..i]));
                out.push(st.id);
                i += st.content.len();
                piece_start = i;
            }
            None => i += s[i..].chars().next().unwrap().len_utf8(),
        }
    }
    out.extend(model.encode(&s.as_bytes()[piece_start..]));
    out
}

#[test]
fn special_tokens_are_split_out_before_bpe_with_native_ids() {
    let wc = word_counts(&corpus_docs());
    let mut model = BottomUpTrainer::new(&BpeScorer, 400).train(&wc).unwrap();
    // Plain ASCII, one containing a space and one non-ASCII: the last two have a byte-level
    // printable form different from their raw content.
    for st in ["<|endoftext|>", "<sep> x", "▁mask▁"] {
        model.vocab.add_special_token(st).unwrap();
    }
    let tok = hf(&HfTokenizerJson::from_bpe_model(&model, None));
    for st in model.vocab.special_tokens() {
        assert_eq!(tok.token_to_id(&st.content), Some(st.id), "special {:?}", st.content);
    }
    for s in [
        "hello<|endoftext|>world",
        "<|endoftext|><|endoftext|>",
        "a <sep> x b<sep> xx",
        "日本語▁mask▁テキスト",
        "no specials here",
    ] {
        assert_eq!(split_specials_then(&model, s), hf_ids(&tok, s), "for {s:?}");
    }
}

#[test]
fn whitespace_ranges_are_exactly_char_is_whitespace() {
    for cp in 0..=0x10FFFFu32 {
        let Some(c) = char::from_u32(cp) else { continue };
        let listed = WHITESPACE_RANGES.iter().any(|&(lo, hi)| lo <= c && c <= hi);
        assert_eq!(listed, c.is_whitespace(), "U+{cp:04X}");
    }
}

#[test]
fn hf_pre_tokenizer_splits_exactly_like_segment_text() {
    let model = &bpe_models()[0].1;
    let tok = hf(&HfTokenizerJson::from_bpe_model(model, None));
    let pre = tok.get_pre_tokenizer().expect("pre_tokenizer");
    let mut probe = String::from("a");
    for &(lo, hi) in &WHITESPACE_RANGES {
        for c in lo..=hi {
            probe.push(c);
            probe.push('b');
        }
    }
    probe.push_str("x\u{200b}\u{feff}\u{180e}\u{1c}\u{1f}\u{ad}y");
    for s in TESTS.iter().copied().chain([probe.as_str()]) {
        let mut pts = PreTokenizedString::from(s);
        pre.pre_tokenize(&mut pts).unwrap();
        let hf_offsets: Vec<(usize, usize)> = pts
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(_, offsets, _)| offsets)
            .collect();
        let mut native_offsets = Vec::new();
        let mut pos = 0;
        for seg in SuperBpeTrainer::segment_bytes(s.as_bytes()) {
            native_offsets.push((pos, pos + seg.len()));
            pos += seg.len();
        }
        assert_eq!(hf_offsets, native_offsets, "for {s:?}");
    }
}

/// With Unicode-whitespace segments a byte-level merge learned inside words can straddle a
/// multi-byte space: here (0xAD, 0xE3) from "中、"-like words would glue "中" to U+3000 (E3 80 80)
/// if text were encoded whole. Training never sees that, so HF must split segments first.
#[test]
fn export_keeps_merges_inside_unicode_whitespace_segments() {
    let words: Vec<String> = ['中', '乭', '亭', '侭', '傭']
        .iter()
        .flat_map(|c| ['、', '。', 'の', 'を', 'に', 'が'].map(move |h| format!("{c}{h}")))
        .collect();
    assert!(words.iter().all(|w| w.as_bytes().windows(2).any(|p| p == [0xAD, 0xE3])));
    let wc: Vec<(Vec<u8>, u64)> = words.iter().map(|w| (w.as_bytes().to_vec(), 10)).collect();
    let model = BottomUpTrainer::new(&BpeScorer, 257).train(&wc).unwrap();
    assert_eq!(model.vocab.get_bytes(256), Some(&[0xAD, 0xE3][..]), "top merge must be (AD, E3)");

    let s = "中\u{3000}中";
    let per_segment: Vec<TokenId> = SuperBpeTrainer::segment_bytes(s.as_bytes())
        .into_iter()
        .flat_map(|seg| model.encode(seg))
        .collect();
    // No merge applies inside either segment, so training-consistent ids are the raw bytes;
    // BPE over the unsplit text would emit token 256 = (AD, E3) straddling U+3000. Native
    // `BottomUpModel::encode` must pre-split the same way when `model.pretokenize` is set.
    assert_eq!(per_segment, s.bytes().map(TokenId::from).collect::<Vec<_>>());
    assert!(model.pretokenize);
    assert_eq!(model.encode(s.as_bytes()), per_segment);
    let tok = hf(&HfTokenizerJson::from_bpe_model(&model, None));
    assert_eq!(hf_ids(&tok, s), per_segment);
}
