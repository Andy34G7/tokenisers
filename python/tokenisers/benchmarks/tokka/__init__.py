"""
Tokka-Bench integration runner for multilingual and programming language tokenizer evaluation.
Computes bytes_per_token, subword_fertility, continued_word_rate, and unique_tokens
across natural languages (FineWeb-2) and programming languages (StarCoder).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
CJK_SEGMENT_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|\w+|[^\s\w]")
CODE_SEGMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|==|!=|<=|>=|->|::|[^\s\w]")


class TokkaBenchRunner:
    """Tokka-bench execution harness supporting natural and programming languages."""

    SUPPORTED_LANGUAGES = {
        "en", "de", "fr", "ru", "zh", "ja", "ar", "hi", "hu", "fi",
        "code", "python", "rust", "cpp", "javascript",
    }

    def get_language_segmenter(self, lang_code: str) -> Callable[[str], List[str]]:
        if lang_code not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {lang_code}")
        if lang_code in {"zh", "ja"}:
            return lambda text: [tok for tok in CJK_SEGMENT_RE.findall(text) if not tok.isspace()]
        if lang_code in {"code", "python", "rust", "cpp", "javascript"}:
            return lambda text: [tok for tok in CODE_SEGMENT_RE.findall(text) if not tok.isspace()]
        return lambda text: [w for w in text.split() if w]

    def _segment_words(self, text_sample: str, lang_code: Optional[str] = None) -> List[str]:
        if lang_code is not None:
            segmenter = self.get_language_segmenter(lang_code)
            words = segmenter(text_sample)
            if words:
                return words
        # Auto-detect unspaced CJK text when lang_code is not explicitly supplied
        if CJK_CHAR_RE.search(text_sample) and len(text_sample.split()) <= max(1, len(text_sample) // 15):
            words = [tok for tok in CJK_SEGMENT_RE.findall(text_sample) if not tok.isspace()]
            if words:
                return words
        return [w for w in text_sample.split() if w]

    def evaluate_tokenizer(
        self,
        tokenizer: Any,
        text_sample: str = "",
        lang_code: Optional[str] = None,
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

        words = self._segment_words(text_sample, lang_code=lang_code)
        num_words = max(len(words), 1)

        bytes_per_token = raw_bytes / num_tokens
        word_token_counts = [len(tokenizer.encode(w, **kwargs)) for w in words]
        subword_fertility = sum(word_token_counts) / num_words
        continued_words = sum(1 for c in word_token_counts if c > 1)
        continued_word_rate = (continued_words / num_words) * 100.0
        unique_tokens = len(set(token_ids))

        return {
            "bytes_per_token": float(bytes_per_token),
            "subword_fertility": float(subword_fertility),
            "continued_word_rate": float(continued_word_rate),
            "unique_tokens": int(unique_tokens),
        }

    def run_and_save(
        self,
        tokenizer: Any,
        text_sample: str,
        output_path: str,
        lang_code: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        metrics = self.evaluate_tokenizer(
            tokenizer, text_sample=text_sample, lang_code=lang_code, **kwargs
        )
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": getattr(tokenizer, "vocab_size", 0),
            "lang_code": lang_code or "auto",
            "metrics": metrics,
            **metrics,
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return metrics

    def evaluate_matrix(
        self,
        models: Dict[str, Any],
        domain_texts: Dict[str, str],
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluates multiple tokenizer variants across multiple language/code domains,
        saving JSON, CSV, and Markdown leaderboards when output_dir is specified.
        """
        matrix: Dict[str, Dict[str, Dict[str, float]]] = {}
        for model_name, tok in models.items():
            matrix[model_name] = {}
            agg_bpt: List[float] = []
            agg_fert: List[float] = []
            agg_cwr: List[float] = []
            agg_uniq: List[float] = []

            for domain_key, text in domain_texts.items():
                lang = domain_key if domain_key in self.SUPPORTED_LANGUAGES else None
                res = self.evaluate_tokenizer(tok, text_sample=text, lang_code=lang)
                matrix[model_name][domain_key] = res
                agg_bpt.append(res["bytes_per_token"])
                agg_fert.append(res["subword_fertility"])
                agg_cwr.append(res["continued_word_rate"])
                agg_uniq.append(float(res["unique_tokens"]))

            n_dom = max(1, len(domain_texts))
            matrix[model_name]["macro_avg"] = {
                "bytes_per_token": sum(agg_bpt) / n_dom,
                "subword_fertility": sum(agg_fert) / n_dom,
                "continued_word_rate": sum(agg_cwr) / n_dom,
                "unique_tokens": int(sum(agg_uniq)),
            }

        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "tokka_matrix.json", "w", encoding="utf-8") as f:
                json.dump(matrix, f, indent=2)

            # Save CSV summary
            csv_path = out_dir / "tokka_leaderboard.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "tokenizer",
                    "domain",
                    "bytes_per_token",
                    "subword_fertility",
                    "continued_word_rate",
                    "unique_tokens",
                ])
                for m_name, dom_map in matrix.items():
                    for d_name, m_vals in dom_map.items():
                        writer.writerow([
                            m_name,
                            d_name,
                            f"{m_vals['bytes_per_token']:.4f}",
                            f"{m_vals['subword_fertility']:.4f}",
                            f"{m_vals['continued_word_rate']:.2f}",
                            int(m_vals["unique_tokens"]),
                        ])

            # Save Markdown leaderboard
            md_lines = [
                "# Tokka-Bench Multilingual & Code Evaluation Leaderboard",
                "",
                "| Tokenizer Variant | Macro Bytes/Token (↑) | Macro Fertility (↓) | Continued Word Rate % (↓) | Total Unique Tokens (↑) |",
                "| :--- | ---: | ---: | ---: | ---: |",
            ]
            sorted_models = sorted(
                matrix.items(),
                key=lambda kv: kv[1]["macro_avg"]["bytes_per_token"],
                reverse=True,
            )
            for m_name, dom_map in sorted_models:
                avg = dom_map["macro_avg"]
                md_lines.append(
                    f"| `{m_name}` | {avg['bytes_per_token']:.3f} | {avg['subword_fertility']:.3f} | "
                    f"{avg['continued_word_rate']:.2f}% | {int(avg['unique_tokens'])} |"
                )
            (out_dir / "tokka_leaderboard.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        return matrix


__all__ = ["TokkaBenchRunner"]
