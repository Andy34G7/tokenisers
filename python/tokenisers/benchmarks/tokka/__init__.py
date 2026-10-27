"""
Tokka-Bench integration runner for tokenizer evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class TokkaBenchRunner:
    """Tokka-bench execution harness."""

    SUPPORTED_LANGUAGES = {"en", "de", "fr", "ru", "zh", "ja", "ar", "hi", "hu", "fi"}

    def get_language_segmenter(self, lang_code: str) -> Any:
        if lang_code not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {lang_code}")
        return lambda text: text.split()

    def evaluate_tokenizer(
        self,
        tokenizer: Any,
        text_sample: str = "",
        **kwargs: Any,
    ) -> Dict[str, float]:
        if not text_sample:
            return {
                "bytes_per_token": 0.0,
                "subword_fertility": 0.0,
                "continued_word_rate": 0.0,
                "unique_tokens": 0,
            }

        raw_bytes = len(text_sample.encode("utf-8"))
        token_ids = tokenizer.encode(text_sample, **kwargs)
        num_tokens = len(token_ids)
        if num_tokens == 0:
            return {
                "bytes_per_token": 0.0,
                "subword_fertility": 0.0,
                "continued_word_rate": 0.0,
                "unique_tokens": 0,
            }

        words = text_sample.split()
        num_words = max(len(words), 1)

        bytes_per_token = raw_bytes / num_tokens
        word_token_counts = [len(tokenizer.encode(w, **kwargs)) for w in words]
        subword_fertility = sum(word_token_counts) / num_words
        continued_words = sum(1 for c in word_token_counts if c > 1)
        continued_word_rate = (continued_words / num_words) * 100.0
        unique_tokens = len(set(token_ids))

        return {
            "bytes_per_token": bytes_per_token,
            "subword_fertility": subword_fertility,
            "continued_word_rate": continued_word_rate,
            "unique_tokens": unique_tokens,
        }

    def run_and_save(
        self,
        tokenizer: Any,
        text_sample: str,
        output_path: str,
        **kwargs: Any,
    ) -> None:
        metrics = self.evaluate_tokenizer(tokenizer, text_sample=text_sample, **kwargs)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, **metrics}, f, indent=2)


__all__ = ["TokkaBenchRunner"]
