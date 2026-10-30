"""
Tokka-Bench metrics for multilingual and programming-language text.

Word-level metrics come from the vendored tokka-bench code (vendor/tokka-bench,
``tokka_bench.metrics.calculate_word_metrics``); bytes/token, total and unique tokens are computed
exactly as its ``fast_benchmark._compute_language_metrics_from_ids`` does (a test checks both
against the vendored functions). Definitions, per language:

- bytes_per_token = UTF-8 bytes / tokens of the whole text
- subword_fertility = tokens of the whole text / units (whitespace words; characters for
  character-based scripts such as Hani/Jpan)
- continued_word_rate = continuation_token_pct: share (%) of tokens that continue a unit, over an
  even sample of up to 1000 units, each encoded on its own
- word_split_pct: share (%) of sampled units split into more than one token
- unique_tokens: distinct ids used on that language's text
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

_VENDOR_SRC = Path(__file__).resolve().parents[4] / "vendor" / "tokka-bench" / "src"
if str(_VENDOR_SRC) not in sys.path:
    sys.path.append(str(_VENDOR_SRC))
from tokka_bench.metrics import _segment_units, calculate_word_metrics  # noqa: E402

CJK_CHAR_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")

# Short codes (result keys) -> tokka-bench (iso_code, script).
SHORT_LANGS: Dict[str, tuple] = {
    "en": ("eng", "Latn"),
    "de": ("deu", "Latn"),
    "fr": ("fra", "Latn"),
    "es": ("spa", "Latn"),
    "ru": ("rus", "Cyrl"),
    "zh": ("cmn", "Hani"),
    "ja": ("jpn", "Jpan"),
    "ar": ("arb", "Arab"),
    "hi": ("hin", "Deva"),
    "hu": ("hun", "Latn"),
    "fi": ("fin", "Latn"),
}
CODE_LANGS = {"code", "python", "rust", "cpp", "javascript", "java", "go"}
_SHORT_BY_FINEWEB = {f"{iso}_{script}": short for short, (iso, script) in SHORT_LANGS.items()}

_WORD_KEYS = ("subword_fertility", "continued_word_rate", "word_split_pct", "continuation_token_pct")


def short_lang(lang: str) -> str:
    """Result key for a dataset language label: eng_Latn -> en, python -> python."""
    return _SHORT_BY_FINEWEB.get(lang, lang)


def language_info(lang_code: str) -> Dict[str, str]:
    """tokka-bench language_info for a short code (en), a FineWeb code (cmn_Hani) or a code language."""
    if lang_code in SHORT_LANGS:
        iso, script = SHORT_LANGS[lang_code]
        return {"iso_code": iso, "script": script}
    if lang_code in CODE_LANGS:
        return {"iso_code": lang_code, "script": "code"}
    m = re.fullmatch(r"([a-z]{3})_([A-Z][a-z]{3})", lang_code)
    if m:
        return {"iso_code": m.group(1), "script": m.group(2)}
    raise ValueError(f"Unsupported language code: {lang_code}")


class _Encoder:
    """tokka-bench tokenizer protocol over a tokenisers model; per-call kwargs (dropout, ...) pass through."""

    def __init__(self, tokenizer: Any, kwargs: Dict[str, Any]) -> None:
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return list(self.tokenizer.encode(text, **self.kwargs))


def language_metrics(
    tokenizer: Any, text: str, info: Optional[Dict[str, str]], **kwargs: Any
) -> Tuple[Dict[str, Any], Set[int]]:
    """(tokka-bench per-language metrics of ``text`` (see module docstring), set of ids used)."""
    enc = _Encoder(tokenizer, kwargs)
    ids = enc.encode(text)
    word = calculate_word_metrics(enc, text, info, pretokenized_text_token_ids=ids)
    text_bytes = len(text.encode("utf-8"))
    return {
        "bytes_per_token": text_bytes / len(ids) if ids else 0.0,
        "total_bytes": text_bytes,
        "total_tokens": len(ids),
        "unique_tokens": len(set(ids)),
        **{k: word[k] for k in _WORD_KEYS},
        "total_units": word["debug_info"]["total_words"],
        "segmentation_method": word["debug_info"]["segmentation_method"],
    }, set(ids)


class TokkaBenchRunner:
    """Tokka-bench execution harness supporting natural and programming languages."""

    SUPPORTED_LANGUAGES = set(SHORT_LANGS) | CODE_LANGS

    def get_language_segmenter(self, lang_code: str) -> Callable[[str], List[str]]:
        info = language_info(lang_code)
        return lambda text: _segment_units(text, info)[0]

    def evaluate_tokenizer(
        self,
        tokenizer: Any,
        text_sample: str = "",
        lang_code: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Tokka-bench metrics of ``text_sample``; ``kwargs`` go to every ``encode`` call. Without
        ``lang_code``, text that is mostly CJK characters is segmented per character (Hani),
        anything else at whitespace."""
        if not text_sample:
            return {"bytes_per_token": 0.0, "total_bytes": 0, "total_tokens": 0, "unique_tokens": 0,
                    **{k: 0.0 for k in _WORD_KEYS}, "total_units": 0}
        if lang_code is not None:
            info: Optional[Dict[str, str]] = language_info(lang_code)
        elif CJK_CHAR_RE.search(text_sample) and len(text_sample.split()) <= max(1, len(text_sample) // 15):
            info = {"iso_code": "cmn", "script": "Hani"}
        else:
            info = None
        return language_metrics(tokenizer, text_sample, info, **kwargs)[0]

    def run_and_save(
        self,
        tokenizer: Any,
        text_sample: str,
        output_path: str,
        lang_code: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        metrics = self.evaluate_tokenizer(tokenizer, text_sample=text_sample, lang_code=lang_code, **kwargs)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": tokenizer.vocab_size,
            "lang_code": lang_code or "auto",
            "metrics": metrics,
        }
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return metrics

    def evaluate_matrix(
        self,
        models: Dict[str, Any],
        domain_texts: Dict[str, str],
        output_dir: Optional[str | Path] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates every tokenizer on every language text (keys: language codes accepted by
        ``language_info``). Per model: one entry per language, ``macro_avg`` (unweighted mean over
        languages), ``micro_avg`` (bytes/token and fertility from totals summed over languages,
        i.e. token-weighted), ``unique_tokens`` = size of the union of ids used across languages,
        and ``vocab_size``. With ``output_dir``, writes tokka_matrix.json ({"meta", "models"}),
        tokka_leaderboard.csv and tokka_leaderboard.md.
        """
        matrix: Dict[str, Dict[str, Any]] = {}
        for model_name, tok in models.items():
            row: Dict[str, Any] = {}
            union: set = set()
            for lang, text in domain_texts.items():
                row[lang], ids = language_metrics(tok, text, language_info(lang))
                union |= ids
            langs = [row[lang] for lang in domain_texts]
            n = max(1, len(langs))
            tokens = sum(r["total_tokens"] for r in langs)
            row["macro_avg"] = {k: sum(r[k] for r in langs) / n for k in ("bytes_per_token", *_WORD_KEYS)}
            row["micro_avg"] = {
                "bytes_per_token": sum(r["total_bytes"] for r in langs) / max(1, tokens),
                "subword_fertility": tokens / max(1, sum(r["total_units"] for r in langs)),
            }
            row["unique_tokens"] = len(union)
            row["vocab_size"] = tok.vocab_size
            matrix[model_name] = row

        if output_dir is not None:
            from tokenisers.benchmarks.report import render_tokka

            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            doc = {"meta": meta or {}, "models": matrix}
            (out_dir / "tokka_matrix.json").write_text(
                json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            with open(out_dir / "tokka_leaderboard.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["tokenizer", "language", "bytes_per_token", "subword_fertility",
                                 "continued_word_rate", "word_split_pct", "unique_tokens", "total_bytes"])
                for m_name, row in matrix.items():
                    for lang in domain_texts:
                        r = row[lang]
                        writer.writerow([m_name, lang, f"{r['bytes_per_token']:.4f}",
                                         f"{r['subword_fertility']:.4f}", f"{r['continued_word_rate']:.2f}",
                                         f"{r['word_split_pct']:.2f}", r["unique_tokens"], r["total_bytes"]])
            (out_dir / "tokka_leaderboard.md").write_text(render_tokka(doc) + "\n", encoding="utf-8")

        return matrix


__all__ = ["TokkaBenchRunner", "language_info", "language_metrics", "short_lang"]
