"""
Pytest configuration and shared fixtures for Tokenisers E2E test suite.
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure python/ and project root are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_PKG_PATH = REPO_ROOT / "python"

if str(PYTHON_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(PYTHON_PKG_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return repository root path."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def colab_cli_path() -> Path:
    """Return path to colab CLI binary."""
    return Path("/home/andey/.local/bin/colab")


@pytest.fixture(scope="session")
def sample_english_text() -> str:
    """Standard English text sample for tokenization tests."""
    return (
        "Tokenization is the foundational step in natural language processing and modern "
        "large language models. High-quality tokenization balances compression efficiency, "
        "subword fertility, and morphological boundary alignment across diverse domains."
    )


@pytest.fixture(scope="session")
def sample_multilingual_text() -> dict:
    """Multilingual text samples across diverse scripts."""
    return {
        "en": "The quick brown fox jumps over the lazy dog.",
        "de": "Falsches Üben von Xylophonmusik quält jeden größeren Zwerg.",
        "fr": "Portez ce vieux whisky au juge blond qui fume.",
        "ru": "Съешь же ещё этих мягких французских булок, да выпей чаю.",
        "zh": "道可道，非常道；名可名，非常名。无名天地之始，有名万物之母。",
        "ja": "いろはにほへと ちりぬるを わかよたれそ つねならむ うゐのおくやま けふこえて あさきゆめみし ゑひもせす",
        "ar": "نص حكيم له سر قاطع وذو شأن عظيم مكتوب على ثوب أخضر ومغلف بجلد غزال.",
        "hi": "ऋषियों को ज्ञान और शांति की खोज में हिमालय के शांत वातावरण में तपस्या करते देखा गया है।",
        "hu": "Egy hűtlen feleség gyönyörű kalandja a varázslatos éjszakában.",
        "fi": "Viekas kettu hypähti laiskan koiran ylitse ja juoksi metsän siimekseen.",
    }


@pytest.fixture(scope="session")
def sample_code_snippets() -> dict:
    """Polyglot source code snippets for code tokenization testing."""
    return {
        "python": (
            "async def fetch_payload(url: str, timeout: float = 30.0) -> dict[str, Any]:\n"
            "    headers = {'Accept': 'application/json', 'User-Agent': 'Tokenisers/1.0'}\n"
            "    async with aiohttp.ClientSession() as session:\n"
            "        async with session.get(url, headers=headers, timeout=timeout) as response:\n"
            "            response.raise_for_status()\n"
            "            return await response.json()\n"
        ),
        "rust": (
            "pub struct TokenizerModel<S: PairScorer> {\n"
            "    vocab: HashMap<TokenId, Vec<u8>>,\n"
            "    merges: Vec<(Pair, TokenId)>,\n"
            "    scorer: S,\n"
            "}\n\n"
            "impl<S: PairScorer> TokenizerModel<S> {\n"
            "    pub fn encode(&self, text: &str) -> Vec<TokenId> {\n"
            "        let mut arena = SymbolArena::from_bytes(text.as_bytes());\n"
            "        arena.apply_merges(&self.merges);\n"
            "        arena.to_token_ids()\n"
            "    }\n"
            "}\n"
        ),
        "cpp": (
            "#include <iostream>\n"
            "#include <vector>\n"
            "template <typename T>\n"
            "class VectorBuffer {\n"
            "private:\n"
            "    std::vector<T> data_;\n"
            "public:\n"
            "    void push(const T& item) { data_.push_back(item); }\n"
            "    size_t size() const noexcept { return data_.size(); }\n"
            "};\n"
        ),
        "javascript": (
            "export const computeRenyiEfficiency = (tokenCounts, alpha = 1.0) => {\n"
            "  const total = tokenCounts.reduce((acc, c) => acc + c, 0);\n"
            "  if (total === 0) return 0.0;\n"
            "  const probs = tokenCounts.map(c => c / total).filter(p => p > 0);\n"
            "  if (alpha === 1.0) {\n"
            "    const entropy = -probs.reduce((acc, p) => acc + p * Math.log2(p), 0);\n"
            "    return entropy / Math.log2(tokenCounts.length);\n"
            "  }\n"
            "  const sumPow = probs.reduce((acc, p) => acc + Math.pow(p, alpha), 0);\n"
            "  const entropy = (1 / (1 - alpha)) * Math.log2(sumPow);\n"
            "  return entropy / Math.log2(tokenCounts.length);\n"
            "};\n"
        ),
    }


@pytest.fixture(scope="session")
def sample_morpheme_annotations() -> list[dict]:
    """Sample gold morphological segmentation pairs (word, gold_cuts)."""
    return [
        {"word": "unhappiness", "segments": ["un", "happi", "ness"], "gold_cuts": {2, 7}},
        {"word": "internationalization", "segments": ["inter", "nation", "al", "iz", "ation"], "gold_cuts": {5, 11, 13, 15}},
        {"word": "reconstruction", "segments": ["re", "construct", "ion"], "gold_cuts": {2, 11}},
        {"word": "disagreement", "segments": ["dis", "agree", "ment"], "gold_cuts": {3, 8}},
        {"word": "pretokenization", "segments": ["pre", "token", "iz", "ation"], "gold_cuts": {3, 8, 10}},
        {"word": "megabyte", "segments": ["mega", "byte"], "gold_cuts": {4}},
        {"word": "subword", "segments": ["sub", "word"], "gold_cuts": {3}},
    ]


@pytest.fixture
def require_tokenisers():
    """Skip test if tokenisers module is not yet compiled or installed, or BpeTokenizer is not implemented."""
    pkg = pytest.importorskip(
        "tokenisers",
        reason="tokenisers package not yet built/installed (M1/M2 implementation in progress)"
    )
    if getattr(pkg, "BpeTokenizer", None) is None:
        pytest.skip("tokenisers.BpeTokenizer not yet implemented (M1 in progress, planned for M2)")
    return pkg
