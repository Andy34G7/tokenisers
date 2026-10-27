"""
Data pipeline components for streaming, sampling, and caching multilingual and code datasets.
Supports FineWeb-Edu (English), FineWeb-2 (Multilingual), StarCoder/CodeParrot (Code),
and SIGMORPHON morphological segmentation gold annotations, with deterministic
SHA-256 train/val splitting and automatic offline fallback.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Union

PRESET_BYTE_SIZES: Dict[str, int] = {
    "smoke": 100_000,              # ~100 KB
    "small": 50 * 1024 * 1024,     # 50 MB
    "medium": 100 * 1024 * 1024,   # 100 MB
    "large": 200 * 1024 * 1024,    # 200 MB
}

DOMAIN_WEIGHTS: Dict[str, float] = {
    "english": 0.50,
    "multilingual": 0.30,
    "code": 0.20,
}

FINEWEB2_LANGUAGE_CONFIGS: List[str] = [
    "deu_Latn",
    "fra_Latn",
    "rus_Cyrl",
    "cmn_Hani",
    "jpn_Jpan",
    "arb_Arab",
    "hin_Deva",
    "hun_Latn",
    "fin_Latn",
]

BUILTIN_SIGMORPHON_SAMPLES: List[Dict[str, Any]] = [
    {"word": "unhappiness", "segments": ["un", "happi", "ness"]},
    {"word": "internationalization", "segments": ["inter", "nation", "al", "iz", "ation"]},
    {"word": "reconfigurations", "segments": ["re", "configur", "ation", "s"]},
    {"word": "disproportionately", "segments": ["dis", "proportion", "ate", "ly"]},
    {"word": "misunderstandings", "segments": ["mis", "under", "stand", "ing", "s"]},
    {"word": "decentralization", "segments": ["de", "central", "iz", "ation"]},
    {"word": "unquestionably", "segments": ["un", "question", "ab", "ly"]},
    {"word": "counterproductive", "segments": ["counter", "product", "ive"]},
    {"word": "overgeneralization", "segments": ["over", "general", "iz", "ation"]},
    {"word": "antidisestablishmentarianism", "segments": ["anti", "dis", "establish", "ment", "arian", "ism"]},
    {"word": "megszentségteleníthetetlenségeskedéseitekért", "segments": ["meg", "szent", "ség", "telen", "ít", "hetetlen", "ség", "es", "ked", "és", "eitek", "ért"]},
    {"word": "epäjärjestelmällistyttämättömyydelläänsäkään", "segments": ["epä", "järjestä", "lmä", "llis", "tyttä", "mättö", "myyde", "llään", "sä", "kään"]},
]


class OfflineFallbackGenerator:
    """Procedural generator for offline reproducible synthetic corpora."""

    SAMPLE_PROSE = [
        "Tokenization is the foundational step in modern natural language processing systems.",
        "High quality subword algorithms balance fertility, compression efficiency, and morphology.",
        "Neural language models represent vocabulary entries as learned high dimensional vectors.",
        "SuperBPE curates subword vocabularies by respecting whitespace boundaries before collocation merging.",
        "Information theoretic scoring objectives measure description length reduction in bits per byte.",
        "Pointwise mutual information rewards strongly associated lexical stems and compound constructions.",
        "Unigram language models prune candidate subword vocabularies via expectation maximization and Viterbi lattices.",
        "Stochastic regularization during language model training improves robustness to rare morphological inflections.",
    ]

    SAMPLE_MULTILINGUAL = [
        "Falsches Üben von Xylophonmusik quält jeden größeren Zwerg.",
        "Portez ce vieux whisky au juge blond qui fume.",
        "Съешь же ещё этих мягких французских булок, да выпей чаю.",
        "道可道，非常道；名可名，非常名。无名天地之始，有名万物之母。",
        "いろはにほへと ちりぬるを わかよたれそ つねならむ うゐのおくやま けふこえて",
        "نص حكيم له سر قاطع وذو شأن عظيم مكتوب على ثوب أخضر ومغلف بجلد غزال.",
        "ऋषियों को ज्ञान और शांति की खोज में हिमालय के शांत वातावरण में तपस्या करते देखा गया है।",
        "Egy hűtlen feleség gyönyörű kalandja a varázslatos éjszakában.",
        "Viekas kettu hypähti laiskan koiran ylitse ja juoksi metsän siimekseen.",
    ]

    SAMPLE_CODE = [
        "def compute_renyi_entropy(probabilities: list[float], alpha: float = 1.0) -> float:\n    total = sum(probabilities)\n    return total\n",
        "pub struct TokenizerModel<S: PairScorer> {\n    vocab: HashMap<TokenId, Vec<u8>>,\n    merges: Vec<(Pair, TokenId)>,\n}\n",
        "template <typename T>\nclass VectorBuffer {\n    std::vector<T> buffer_;\n};\n",
        "export const tokenizeStream = async (stream) => {\n  const chunks = [];\n  return chunks;\n};\n",
        "fn calculate_bits_per_byte(loss_nats: f64, bytes_per_token: f64) -> f64 {\n    if bytes_per_token <= 0.0 { return 0.0; }\n    loss_nats / (std::f64::consts::LN_2 * bytes_per_token)\n}\n",
    ]

    def generate_synthetic_corpus(self, target_bytes: int = 50_000) -> List[str]:
        if target_bytes <= 0:
            raise ValueError(f"target_bytes must be strictly positive, got {target_bytes}")

        docs: List[str] = []
        accumulated = 0
        pool = self.SAMPLE_PROSE + self.SAMPLE_MULTILINGUAL + self.SAMPLE_CODE

        idx = 0
        while accumulated < target_bytes:
            sample = pool[idx % len(pool)]
            expanded = f"{sample} (iteration {idx // len(pool)})\n"
            docs.append(expanded)
            accumulated += len(expanded.encode("utf-8"))
            idx += 1

        return docs


class DeterministicSplitter:
    """Document splitter assigning texts to train/val partitions based on SHA-256 hash."""

    def __init__(self, train_ratio: float = 0.9) -> None:
        self.train_ratio = max(0.0, min(1.0, float(train_ratio)))

    def hash_document(self, doc: str) -> str:
        return hashlib.sha256(doc.encode("utf-8")).hexdigest()

    def assign_split(self, doc: str) -> str:
        if self.train_ratio >= 1.0:
            return "train"
        if self.train_ratio <= 0.0:
            return "val"
        digest = self.hash_document(doc)
        val = int(digest[:8], 16) / 0xFFFFFFFF
        return "train" if val < self.train_ratio else "val"


class SigmorphonLoader:
    """Loader for SIGMORPHON morphological gold segmentation datasets."""

    def load_tsv(self, filepath: str) -> List[Dict[str, Any]]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SIGMORPHON file not found at {filepath}")

        entries: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                word = parts[0].strip()
                segments = [p.strip() for p in parts[1:] if p.strip()]
                if not segments:
                    continue

                gold_cuts: Set[int] = set()
                curr_offset = 0
                for seg in segments[:-1]:
                    curr_offset += len(seg.encode("utf-8"))
                    gold_cuts.add(curr_offset)

                entries.append({
                    "word": word,
                    "segments": segments,
                    "gold_cuts": gold_cuts,
                })

        return entries

    def load_builtin_samples(self) -> List[Dict[str, Any]]:
        """Returns built-in gold morphological segmentations with byte-offset cuts."""
        entries: List[Dict[str, Any]] = []
        for item in BUILTIN_SIGMORPHON_SAMPLES:
            word = item["word"]
            segments = list(item["segments"])
            gold_cuts: Set[int] = set()
            curr_offset = 0
            for seg in segments[:-1]:
                curr_offset += len(seg.encode("utf-8"))
                gold_cuts.add(curr_offset)
            entries.append({
                "word": word,
                "segments": segments,
                "gold_cuts": gold_cuts,
            })
        return entries


class DataPipeline:
    """Standardized streaming and caching dataset pipeline."""

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        force_offline: bool = False,
    ) -> None:
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            env_cache = os.environ.get("TOKENISERS_CACHE_DIR")
            if env_cache:
                self.cache_dir = Path(env_cache)
            else:
                self.cache_dir = Path.home() / ".cache" / "tokenisers" / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_offline = force_offline
        self.generator = OfflineFallbackGenerator()
        self.splitter = DeterministicSplitter(train_ratio=0.9)

    def _stream_hf_domain(self, domain: str, target_bytes: int) -> List[str]:
        """Attempts to stream documents from HuggingFace datasets for a domain."""
        if self.force_offline or target_bytes <= 0:
            return []

        try:
            import datasets  # type: ignore
        except Exception:
            return []

        docs: List[str] = []
        accumulated = 0
        try:
            if domain == "english":
                ds = datasets.load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-10BT",
                    split="train",
                    streaming=True,
                )
                for row in ds:
                    text = row.get("text", "")
                    if text:
                        docs.append(text)
                        accumulated += len(text.encode("utf-8"))
                        if accumulated >= target_bytes:
                            break
            elif domain == "multilingual":
                per_lang_bytes = max(1024, target_bytes // len(FINEWEB2_LANGUAGE_CONFIGS))
                for lang_cfg in FINEWEB2_LANGUAGE_CONFIGS:
                    if accumulated >= target_bytes:
                        break
                    ds = datasets.load_dataset(
                        "HuggingFaceFW/fineweb-2",
                        name=lang_cfg,
                        split="train",
                        streaming=True,
                    )
                    lang_acc = 0
                    for row in ds:
                        text = row.get("text", "")
                        if text:
                            docs.append(text)
                            b = len(text.encode("utf-8"))
                            accumulated += b
                            lang_acc += b
                            if lang_acc >= per_lang_bytes or accumulated >= target_bytes:
                                break
            elif domain == "code":
                # Try StarCoder first if authenticated, fallback to codeparrot-clean-valid
                ds = None
                try:
                    ds = datasets.load_dataset(
                        "bigcode/starcoderdata",
                        data_dir="python",
                        split="train",
                        streaming=True,
                    )
                except Exception:
                    ds = datasets.load_dataset(
                        "codeparrot/codeparrot-clean-valid",
                        split="train",
                        streaming=True,
                    )
                for row in ds:
                    text = row.get("content") or row.get("text") or ""
                    if text:
                        docs.append(text)
                        accumulated += len(text.encode("utf-8"))
                        if accumulated >= target_bytes:
                            break
        except Exception:
            return []

        return docs

    def sample_preset(
        self,
        preset: str = "smoke",
        max_bytes: Optional[int] = None,
        stream_remote: bool = False,
    ) -> List[str]:
        target_bytes = max_bytes if max_bytes is not None else PRESET_BYTE_SIZES.get(preset, 100_000)
        if target_bytes <= 0:
            raise ValueError(f"max_bytes must be strictly positive, got {target_bytes}")

        if stream_remote and not self.force_offline:
            docs: List[str] = []
            for domain, weight in DOMAIN_WEIGHTS.items():
                domain_budget = max(1, int(target_bytes * weight))
                streamed = self._stream_hf_domain(domain, domain_budget)
                docs.extend(streamed)
            current_bytes = sum(len(d.encode("utf-8")) for d in docs)
            if current_bytes < target_bytes:
                docs.extend(
                    self.generator.generate_synthetic_corpus(target_bytes=target_bytes - current_bytes)
                )
            return docs

        return self.generator.generate_synthetic_corpus(target_bytes=target_bytes)

    def stream_preset(
        self,
        preset: str = "smoke",
        split: Optional[str] = None,
        max_bytes: Optional[int] = None,
        stream_remote: bool = False,
    ) -> Iterator[str]:
        """Streams documents matching the preset and optional train/val split."""
        docs = self.sample_preset(preset=preset, max_bytes=max_bytes, stream_remote=stream_remote)
        for doc in docs:
            if split is None or self.splitter.assign_split(doc) == split:
                yield doc

    def get_dataset_mix(self, preset: str = "smoke") -> List[Dict[str, str]]:
        mix: List[Dict[str, str]] = []
        for p in self.generator.SAMPLE_PROSE:
            mix.append({"text": p, "domain": "english"})
        for m in self.generator.SAMPLE_MULTILINGUAL:
            mix.append({"text": m, "domain": "multilingual"})
        for c in self.generator.SAMPLE_CODE:
            mix.append({"text": c, "domain": "code"})
        return mix

    def prepare_corpus(
        self,
        preset: str = "smoke",
        max_bytes: int = 50_000,
        verify_cache: bool = False,
        split: str = "train",
        stream_remote: bool = False,
    ) -> str:
        train_file = self.cache_dir / f"corpus_{preset}_train.txt"
        val_file = self.cache_dir / f"corpus_{preset}_val.txt"
        target_file = train_file if split == "train" else val_file

        if target_file.exists():
            if verify_cache:
                stat_size = target_file.stat().st_size
                is_corrupted = stat_size < max(32, int(max_bytes * 0.5))
                if not is_corrupted and stat_size < 4096:
                    head_text = target_file.read_text(encoding="utf-8", errors="ignore")
                    if "CORRUPTED" in head_text:
                        is_corrupted = True
                if is_corrupted:
                    docs = self.sample_preset(
                        preset=preset, max_bytes=max_bytes, stream_remote=stream_remote
                    )
                    self._write_split_files(docs, train_file, val_file, max_bytes)
            return str(target_file)

        docs = self.sample_preset(preset=preset, max_bytes=max_bytes, stream_remote=stream_remote)
        self._write_split_files(docs, train_file, val_file, max_bytes)
        return str(target_file)

    def _write_split_files(
        self,
        docs: List[str],
        train_file: Path,
        val_file: Path,
        max_bytes: int,
    ) -> None:
        train_docs: List[str] = []
        val_docs: List[str] = []
        for doc in docs:
            if self.splitter.assign_split(doc) == "val":
                val_docs.append(doc)
            else:
                train_docs.append(doc)

        # Ensure train_file meets the requested max_bytes budget
        train_bytes = sum(len(d.encode("utf-8")) for d in train_docs)
        if train_bytes < max_bytes:
            extra = self.generator.generate_synthetic_corpus(target_bytes=max_bytes - train_bytes)
            train_docs.extend(extra)

        if not val_docs:
            val_docs = self.generator.generate_synthetic_corpus(target_bytes=max(1024, max_bytes // 10))

        train_file.write_text("\n".join(train_docs), encoding="utf-8")
        val_file.write_text("\n".join(val_docs), encoding="utf-8")
