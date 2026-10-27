"""
Data pipeline components for streaming, sampling, and caching multilingual and code datasets.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class OfflineFallbackGenerator:
    """Procedural generator for offline reproducible synthetic corpora."""

    SAMPLE_PROSE = [
        "Tokenization is the foundational step in modern natural language processing systems.",
        "High quality subword algorithms balance fertility, compression efficiency, and morphology.",
        "Neural language models represent vocabulary entries as learned high dimensional vectors.",
        "SuperBPE curates subword vocabularies by respecting whitespace boundaries before collocation merging.",
        "Information theoretic scoring objectives measure description length reduction in bits per byte.",
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
            self.cache_dir = Path.home() / ".cache" / "tokenisers" / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_offline = force_offline
        self.generator = OfflineFallbackGenerator()
        self.splitter = DeterministicSplitter(train_ratio=0.9)

    def sample_preset(self, preset: str = "smoke", max_bytes: int = 100_000) -> List[str]:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be strictly positive, got {max_bytes}")
        return self.generator.generate_synthetic_corpus(target_bytes=max_bytes)

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
    ) -> str:
        cache_file = self.cache_dir / f"corpus_{preset}_train.txt"

        if cache_file.exists():
            if verify_cache:
                content = cache_file.read_text(encoding="utf-8")
                if "CORRUPTED" in content or len(content.encode("utf-8")) < max_bytes:
                    docs = self.sample_preset(preset=preset, max_bytes=max_bytes)
                    cache_file.write_text("\n".join(docs), encoding="utf-8")
            return str(cache_file)

        docs = self.sample_preset(preset=preset, max_bytes=max_bytes)
        cache_file.write_text("\n".join(docs), encoding="utf-8")
        return str(cache_file)
