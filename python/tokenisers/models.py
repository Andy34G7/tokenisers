"""
High-level Object-Oriented Tokenizer implementations for Tokenisers platform.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Union

from tokenisers import _core


class BpeTokenizer:
    """
    High-level Byte-Level BPE Tokenizer supporting pluggable scoring objectives,
    merge dropout, lossless reconstruction, and HuggingFace export.
    """

    def __init__(self, model: Optional[_core.PyBpeModel] = None) -> None:
        self._model = model

    @property
    def vocab_size(self) -> int:
        if self._model is None:
            return 0
        return self._model.vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        if self._model is None:
            return {}
        return self._model.get_vocab()

    def token_to_id(self, token: str) -> Optional[int]:
        if self._model is None:
            return None
        return self._model.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        if self._model is None:
            return None
        return self._model.id_to_token(token_id)

    def id_to_bytes(self, token_id: int) -> Optional[bytes]:
        if self._model is None:
            return None
        raw = self._model.id_to_bytes(token_id)
        return bytes(raw) if raw is not None else None

    @classmethod
    def train(
        cls,
        corpus: List[str],
        vocab_size: int = 300,
        scorer: str = "frequency",
        min_frequency: int = 0,
        alpha: float = 1.0,
        beta: float = 1.0,
        penalty: float = 0.0,
        special_tokens: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BpeTokenizer:
        if not corpus:
            raise ValueError("Training corpus cannot be empty")
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")

        if "entry_cost" in kwargs:
            penalty = float(kwargs["entry_cost"])

        if "beta" in kwargs:
            b_val = kwargs["beta"]
            if not isinstance(b_val, (int, float)) or (isinstance(b_val, float) and math.isnan(b_val)):
                raise TypeError(f"beta must be a valid float, got {b_val}")
            beta = float(b_val)
        else:
            if not isinstance(beta, (int, float)) or (isinstance(beta, float) and math.isnan(beta)):
                raise TypeError(f"beta must be a valid float, got {beta}")
            beta = float(beta)

        model = _core.train_bpe(
            corpus=corpus,
            vocab_size=vocab_size,
            scorer_type=scorer,
            min_frequency=min_frequency,
            alpha=float(alpha),
            beta=float(beta),
            penalty=float(penalty),
            special_tokens=special_tokens,
        )
        return cls(model=model)

    def encode(
        self,
        text: str,
        dropout: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[int]:
        if self._model is None:
            raise RuntimeError("Tokenizer is not trained or initialized")
        if text == "":
            return []
        if dropout is not None:
            if not isinstance(dropout, (int, float)) or not (0.0 <= float(dropout) <= 1.0) or math.isnan(float(dropout)):
                raise ValueError(f"dropout must be between 0.0 and 1.0, got {dropout}")
            return self._model.encode(text, dropout=float(dropout), seed=seed)
        return self._model.encode(text, dropout=None, seed=seed)

    def decode(self, token_ids: List[int], **kwargs: Any) -> str:
        if self._model is None:
            raise RuntimeError("Tokenizer is not trained or initialized")
        if not token_ids:
            return ""
        return self._model.decode(token_ids)

    def export_hf_json(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Tokenizer is not trained or initialized")
        self._model.export_hf_json(str(path))

    def save_pretrained(self, save_directory: str) -> None:
        if self._model is None:
            raise RuntimeError("Tokenizer is not trained or initialized")
        os.makedirs(save_directory, exist_ok=True)
        json_path = os.path.join(save_directory, "tokenizer.json")
        self.export_hf_json(json_path)
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "model_type": "bpe",
                    "clean_up_tokenization_spaces": False,
                },
                f,
                indent=2,
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> BpeTokenizer:
        model = _core.PyBpeModel.from_hf_json(str(pretrained_model_name_or_path))
        return cls(model=model)


class WordPieceTokenizer(BpeTokenizer):
    """BPE Tokenizer configured with Pointwise Mutual Information (WordPiece/PMI) scoring."""

    @classmethod
    def train(cls, corpus: List[str], vocab_size: int = 300, **kwargs: Any) -> WordPieceTokenizer:
        return super().train(corpus=corpus, vocab_size=vocab_size, scorer="wordpiece", **kwargs)


class MdlTokenizer(BpeTokenizer):
    """BPE Tokenizer configured with Minimum Description Length (MDL / Delta-Bits) scoring."""

    @classmethod
    def train(cls, corpus: List[str], vocab_size: int = 300, **kwargs: Any) -> MdlTokenizer:
        return super().train(corpus=corpus, vocab_size=vocab_size, scorer="mdl", **kwargs)


class LengthPenalizedTokenizer(BpeTokenizer):
    """BPE Tokenizer configured with Length-Penalized scoring."""

    @classmethod
    def train(
        cls, corpus: List[str], vocab_size: int = 300, beta: float = 0.5, **kwargs: Any
    ) -> LengthPenalizedTokenizer:
        return super().train(
            corpus=corpus, vocab_size=vocab_size, scorer="length_penalized", beta=beta, **kwargs
        )


class SuperBpeTokenizer(BpeTokenizer):
    """Two-Stage SuperBPE Tokenizer: word-constrained up to v_switch, then cross-word collocations."""

    @classmethod
    def train(
        cls,
        corpus: List[str],
        v_switch: int = 280,
        v_final: int = 320,
        **kwargs: Any,
    ) -> SuperBpeTokenizer:
        if not corpus:
            raise ValueError("Training corpus cannot be empty")
        if v_switch < 256:
            raise ValueError(f"v_switch must be >= 256, got {v_switch}")
        if v_switch > v_final:
            raise ValueError(f"v_switch ({v_switch}) cannot exceed v_final ({v_final})")
        model = _core.train_super_bpe(corpus, v_switch, v_final)
        return cls(model=model)


class MorphemeGuidedTokenizer(BpeTokenizer):
    """BPE Tokenizer regularized by morphological boundary priors."""

    @classmethod
    def train(
        cls,
        corpus: List[str],
        vocab_size: int = 300,
        morpheme_lexicon: Optional[Dict[str, List[str]]] = None,
        lambda_weight: float = 1.0,
        scorer: str = "frequency",
        **kwargs: Any,
    ) -> MorphemeGuidedTokenizer:
        if not corpus:
            raise ValueError("Training corpus cannot be empty")
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256, got {vocab_size}")
        if morpheme_lexicon is None:
            morpheme_lexicon = {}
        model = _core.train_morpheme_guided(
            corpus, vocab_size, morpheme_lexicon, float(lambda_weight), scorer
        )
        return cls(model=model)


MorphemeBpeTokenizer = MorphemeGuidedTokenizer


class UnigramTokenizer:
    """
    Top-Down Unigram Language Model Tokenizer supporting EM training,
    loss-based vocabulary pruning, Viterbi 1-best decoding, and stochastic lattice sampling.
    """

    def __init__(self, model: Optional[_core.PyUnigramModel] = None) -> None:
        self._model = model

    @property
    def vocab_size(self) -> int:
        if self._model is None:
            return 0
        return self._model.vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        if self._model is None:
            return {}
        return self._model.get_vocab()

    def get_log_probs(self) -> Dict[str, float]:
        if self._model is None:
            return {}
        return self._model.get_log_probs()

    @classmethod
    def train(
        cls,
        corpus: List[str],
        vocab_size: int = 300,
        num_iterations: int = 10,
        shrinking_factor: float = 0.75,
        **kwargs: Any,
    ) -> UnigramTokenizer:
        if not corpus:
            raise ValueError("Training corpus cannot be empty")
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
        model = _core.train_unigram(
            corpus=corpus,
            vocab_size=vocab_size,
            num_iterations=num_iterations,
            shrinking_factor=float(shrinking_factor),
        )
        return cls(model=model)

    def encode(
        self,
        text: str,
        sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        nbest_size: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[int]:
        if self._model is None:
            raise RuntimeError("UnigramTokenizer is not trained or initialized")
        if text == "":
            return []
        if temperature is not None:
            if not isinstance(temperature, (int, float)) or float(temperature) < 0.0 or math.isnan(float(temperature)):
                raise ValueError(f"temperature must be non-negative, got {temperature}")
            temp_val: Optional[float] = float(temperature)
        else:
            temp_val = None

        return self._model.encode(
            text,
            temperature=temp_val,
            nbest_size=nbest_size,
            sample=sample,
            seed=seed,
        )

    def decode(self, token_ids: List[int], **kwargs: Any) -> str:
        if self._model is None:
            raise RuntimeError("UnigramTokenizer is not trained or initialized")
        if not token_ids:
            return ""
        return self._model.decode(token_ids)

    def export_hf_json(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("UnigramTokenizer is not trained or initialized")
        self._model.export_hf_json(str(path))

    def save_pretrained(self, save_directory: str) -> None:
        if self._model is None:
            raise RuntimeError("UnigramTokenizer is not trained or initialized")
        os.makedirs(save_directory, exist_ok=True)
        json_path = os.path.join(save_directory, "tokenizer.json")
        self.export_hf_json(json_path)
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "model_type": "unigram",
                    "clean_up_tokenization_spaces": False,
                },
                f,
                indent=2,
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> UnigramTokenizer:
        model = _core.PyUnigramModel.from_hf_json(str(pretrained_model_name_or_path))
        return cls(model=model)


class ByteEntropyTokenizer:
    """
    Dynamic Byte-Entropy Dynamic Patching Tokenizer based on Markov transition surprise
    and predictive entropy thresholds with min_patch_len and max_patch_len guardrails.
    """

    def __init__(
        self,
        entropy_threshold: float = 4.2,
        min_patch_len: int = 2,
        max_patch_len: int = 16,
        context_order: int = 3,
        smoothing_alpha: float = 0.05,
    ) -> None:
        if not isinstance(entropy_threshold, (int, float)) or float(entropy_threshold) <= 0.0 or math.isnan(float(entropy_threshold)):
            raise ValueError(f"entropy_threshold must be strictly positive, got {entropy_threshold}")

        self.entropy_threshold = float(entropy_threshold)
        self.min_patch_len = int(min_patch_len)
        self.max_patch_len = int(max_patch_len)

        self._patcher = _core.PyByteEntropyPatcher(
            entropy_threshold=self.entropy_threshold,
            context_order=context_order,
            min_len=self.min_patch_len,
            max_len=self.max_patch_len,
            smoothing_alpha=smoothing_alpha,
        )

        self._patch_to_id: Dict[bytes, int] = {}
        self._id_to_patch: Dict[int, bytes] = {}

        # Prepopulate base 256 bytes
        for b in range(256):
            bv = bytes([b])
            self._patch_to_id[bv] = b
            self._id_to_patch[b] = bv

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_patch)

    def _get_or_create_id(self, patch: bytes) -> int:
        if patch in self._patch_to_id:
            return self._patch_to_id[patch]
        new_id = len(self._id_to_patch)
        self._patch_to_id[patch] = new_id
        self._id_to_patch[new_id] = patch
        return new_id

    def encode(self, text: str) -> List[int]:
        if text == "":
            return []
        raw_bytes = text.encode("utf-8")
        patches = self._patcher.segment_stream(raw_bytes)
        return [self._get_or_create_id(bytes(p)) for p in patches]

    def encode_bytes(self, raw_bytes: bytes) -> List[int]:
        if len(raw_bytes) == 0:
            return []
        patches = self._patcher.segment_stream(raw_bytes)
        return [self._get_or_create_id(bytes(p)) for p in patches]

    def id_to_bytes(self, patch_id: int) -> bytes:
        if patch_id in self._id_to_patch:
            return self._id_to_patch[patch_id]
        return bytes([patch_id % 256])

    def decode(self, patch_ids: List[int]) -> str:
        if not patch_ids:
            return ""
        raw = b"".join(self.id_to_bytes(pid) for pid in patch_ids)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")
