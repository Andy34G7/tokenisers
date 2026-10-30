"""
Hugging Face ``PreTrainedTokenizer`` adapter for tokenisers models.

Token strings are the underlying tokenizer's (GPT-2 byte-level printable form, special tokens by
content), so ids round-trip exactly. Padding, truncation, tensors, pairs and batch decoding are the
``PreTrainedTokenizer`` base behaviour. As in GPT-2 (and the exported tokenizer.json, which has
no post-processor), no special-token template exists: ``add_special_tokens=True`` adds nothing,
and bos/eos must be written into the text. Per-call encode options (``dropout``, ``seed``,
``temperature``, ``nbest_size``, ``sample``) are forwarded to the underlying ``encode``. Special
tokens not in the underlying vocabulary get ids from ``vocab_size`` upwards.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

_ENCODE_OPTIONS = ("dropout", "seed", "temperature", "nbest_size", "sample")


def _load_underlying(vocab_file: str):
    from tokenisers.models import BpeTokenizer, UnigramTokenizer

    with open(vocab_file, encoding="utf-8") as f:
        model_type = json.load(f)["model"]["type"]
    classes = {"BPE": BpeTokenizer, "Unigram": UnigramTokenizer}
    if model_type not in classes:
        raise ValueError(f"{vocab_file}: unsupported model type {model_type!r}")
    return classes[model_type].from_pretrained(vocab_file)


class DynamicResearchTokenizer(PreTrainedTokenizer):
    """``PreTrainedTokenizer`` over a trained tokenisers tokenizer (or a saved ``tokenizer.json``)."""

    vocab_files_names = {"vocab_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        vocab_file: Optional[str] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        **kwargs: Any,
    ) -> None:
        if tokenizer is None:
            if vocab_file is None:
                raise ValueError("DynamicResearchTokenizer needs a trained tokenizer or a tokenizer.json vocab_file")
            tokenizer = _load_underlying(vocab_file)
        self._underlying = tokenizer
        # Byte-level decoding is exact; HF's space clean-up would corrupt it.
        kwargs["clean_up_tokenization_spaces"] = False
        super().__init__(unk_token=unk_token, pad_token=pad_token, bos_token=bos_token, eos_token=eos_token, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._underlying.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        vocab = self._underlying.get_vocab()
        vocab.update(getattr(self, "_added_tokens_encoder", {}))
        return vocab

    def tokenize(self, text: str, **kwargs: Any) -> List[str]:
        """Splits out added tokens, then encodes each remaining piece with the per-call options."""
        options = {k: kwargs.pop(k) for k in _ENCODE_OPTIONS if k in kwargs}
        split_special_tokens = kwargs.pop("split_special_tokens", self.split_special_tokens)
        text, _ = self.prepare_for_tokenization(text, **kwargs)  # HF passes e.g. return_offsets_mapping
        if split_special_tokens:
            return self._tokenize(text, **options)
        tokens: List[str] = []
        for piece in self.tokens_trie.split(text):
            if piece in self._added_tokens_encoder:
                tokens.append(piece)
            elif piece:
                tokens.extend(self._tokenize(piece, **options))
        return tokens

    def _tokenize(self, text: str, **options: Any) -> List[str]:
        return [self._underlying.id_to_token(i) for i in self._underlying.encode(text, **options)]

    def _convert_token_to_id(self, token: str) -> int:
        token_id = self._underlying.token_to_id(token)
        return self.unk_token_id if token_id is None else token_id

    def _convert_id_to_token(self, index: int) -> str:
        token = self._underlying.id_to_token(index)
        if token is None:
            raise ValueError(f"Invalid token ID: {index}")
        return token

    def _id_bytes(self, index: int) -> bytes:
        data = self._underlying.id_to_bytes(index)
        if data is not None:
            return data
        if index in self._added_tokens_decoder:
            return self._added_tokens_decoder[index].content.encode("utf-8")
        raise ValueError(f"Invalid token ID: {index}")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ids = [self._convert_token_to_id_with_added_voc(t) for t in tokens]
        return b"".join(self._id_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def _decode(self, token_ids, skip_special_tokens: bool = False, **kwargs: Any) -> str:
        ids = [token_ids] if isinstance(token_ids, int) else list(token_ids)
        if skip_special_tokens:
            special = set(self.all_special_ids)
            ids = [i for i in ids if i not in special]
        return b"".join(self._id_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        os.makedirs(save_directory, exist_ok=True)
        name = f"{filename_prefix}-tokenizer.json" if filename_prefix else "tokenizer.json"
        path = os.path.join(save_directory, name)
        self._underlying.export_hf_json(path)
        return (path,)
