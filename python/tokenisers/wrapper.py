"""
Hugging Face PreTrainedTokenizer-compatible dynamic research tokenizer wrapper.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import BatchEncoding, PreTrainedTokenizer


class DynamicResearchTokenizer(PreTrainedTokenizer):
    """
    Subclass of transformers.PreTrainedTokenizer providing seamless integration
    with PyTorch DataLoader and HuggingFace pipelines while supporting dynamic
    inference parameters (dropout, temperature, nbest_size) and lossless byte reconstruction.
    """

    vocab_files_names = {"vocab_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        clean_up_tokenization_spaces: bool = False,
        **kwargs: Any,
    ) -> None:
        self._underlying = tokenizer
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )
        self.clean_up_tokenization_spaces = False

    @property
    def vocab_size(self) -> int:
        if self._underlying is not None:
            return self._underlying.vocab_size
        return 256

    def get_vocab(self) -> Dict[str, int]:
        if self._underlying is not None:
            return self._underlying.get_vocab()
        return {}

    def _tokenize(self, text: str, **kwargs: Any) -> List[str]:
        if self._underlying is None or text == "":
            return []
        ids = self._underlying.encode(text, **kwargs)
        return [str(i) for i in ids]

    def _convert_token_to_id(self, token: str) -> int:
        if self._underlying is not None:
            val = self._underlying.token_to_id(token)
            if val is not None:
                return val
        if token == self.pad_token:
            return self.pad_token_id if self.pad_token_id is not None else 0
        if token == self.unk_token:
            return self.unk_token_id if self.unk_token_id is not None else 0
        try:
            return int(token)
        except ValueError:
            return 0

    def _convert_id_to_token(self, index: int) -> str:
        if self._underlying is not None:
            tok = self._underlying.id_to_token(index)
            if tok is not None:
                return tok
        return str(index)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: Any,
    ) -> str:
        if self._underlying is None or not token_ids:
            return ""
        clean_ids = token_ids
        if skip_special_tokens:
            special_ids = set(self.all_special_ids)
            clean_ids = [tid for tid in token_ids if tid not in special_ids]
        return self._underlying.decode(clean_ids)

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]], torch.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: Any,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if not token_ids:
            return ""
        if isinstance(token_ids[0], list):
            return " ".join(self._decode(ids, skip_special_tokens=skip_special_tokens, **kwargs) for ids in token_ids)
        return self._decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Any] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], BatchEncoding]:
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        pad_to_multiple_of = kwargs.get("pad_to_multiple_of")

        if isinstance(text, str):
            if text == "" and not add_special_tokens:
                input_ids: List[int] = []
            else:
                input_ids = self._underlying.encode(text, **kwargs) if self._underlying else []
                if truncation and truncation != "do_not_truncate" and max_length is not None:
                    input_ids = input_ids[:max_length]

            orig_len = len(input_ids)
            target_len = orig_len
            if (padding == "max_length" or (padding is True and max_length is not None)) and max_length is not None:
                target_len = max(target_len, max_length)

            if pad_to_multiple_of is not None and pad_to_multiple_of > 0:
                if target_len % pad_to_multiple_of != 0:
                    target_len = ((target_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

            deficit = max(0, target_len - orig_len)
            padded_ids = input_ids + [pad_id] * deficit
            attention_mask = [1] * orig_len + [0] * deficit

            if return_tensors == "pt":
                return BatchEncoding({
                    "input_ids": torch.tensor([padded_ids], dtype=torch.long),
                    "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
                })
            return BatchEncoding({
                "input_ids": padded_ids,
                "attention_mask": attention_mask,
            })

        batch_ids: List[List[int]] = []
        for s in text:
            if s == "" and not add_special_tokens:
                s_ids = []
            else:
                s_ids = self._underlying.encode(s, **kwargs) if self._underlying else []
                if truncation and truncation != "do_not_truncate" and max_length is not None:
                    s_ids = s_ids[:max_length]
            batch_ids.append(s_ids)

        if padding:
            target_len = max((len(x) for x in batch_ids), default=0)
            if padding == "max_length" and max_length is not None:
                target_len = max(target_len, max_length)

            if pad_to_multiple_of is not None and pad_to_multiple_of > 0:
                if target_len % pad_to_multiple_of != 0:
                    target_len = ((target_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

            padded_batch: List[List[int]] = []
            mask_batch: List[List[int]] = []
            for item in batch_ids:
                deficit = max(0, target_len - len(item))
                padded_batch.append(item + [pad_id] * deficit)
                mask_batch.append([1] * len(item) + [0] * deficit)
            batch_ids = padded_batch
            batch_masks = mask_batch
        else:
            batch_masks = [[1] * len(x) for x in batch_ids]

        if return_tensors == "pt":
            return BatchEncoding({
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
            })
        return BatchEncoding({
            "input_ids": batch_ids,
            "attention_mask": batch_masks,
        })

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        os.makedirs(save_directory, exist_ok=True)
        filename = "tokenizer.json"
        if filename_prefix:
            filename = f"{filename_prefix}-{filename}"
        json_path = os.path.join(save_directory, filename)
        if self._underlying is not None:
            self._underlying.export_hf_json(json_path)
        return (json_path,)

    def save_pretrained(
        self, save_directory: str, legacy_format: Optional[bool] = None, filename_prefix: Optional[str] = None, **kwargs: Any
    ) -> Tuple[str, ...]:
        os.makedirs(save_directory, exist_ok=True)
        json_path = os.path.join(save_directory, "tokenizer.json")
        if self._underlying is not None:
            self._underlying.export_hf_json(json_path)
        super().save_pretrained(save_directory, legacy_format=legacy_format, filename_prefix=filename_prefix, **kwargs)
        return (json_path,)
