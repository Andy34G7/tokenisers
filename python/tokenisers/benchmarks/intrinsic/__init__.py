"""
Extended Intrinsic Tokenizer Evaluation Suite.

Rényi efficiency, vocabulary utilisation, morphological and identifier boundary F1, number
fertility and throughput. Boundary offsets always come from the raw bytes of each token
(``id_to_bytes``), so every tokenizer is scored the same way.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

# Fallback when no held-out code is given. The CLI mines identifiers from the test split instead.
DEFAULT_CODE_IDENTIFIERS: List[str] = [
    "getUserProfile",
    "get_user_profile",
    "calculate_loss_v2",
    "HttpRequestHandler",
    "max_retries_count",
]

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+[0-9]*|[A-Z]+[0-9]*|[0-9]+")


def extract_identifier_cuts(ident: str) -> Set[int]:
    """
    Gold boundaries of a code identifier, as byte offsets.

    Rule: the identifier is split into parts at ``_``/``-`` runs, lower->Upper humps
    (``getUser`` -> get|User) and acronym ends (``HTTPServer`` -> HTTP|Server); digits stay with
    the word they follow (``int64``, ``calculate_loss_v2`` -> calculate|loss|v2). A boundary is the
    start offset of every part except the first. Delimiter runs belong to no part: a predicted cut
    anywhere from the start to the end of an inner run counts as the boundary at the run's end, and
    cuts next to leading/trailing runs are ignored (see ``canonical_identifier_cuts``), so splitting
    ``_`` off as its own token is neither rewarded nor penalised.
    """
    starts = [m.start() for m in _PART_RE.finditer(ident)][1:]
    return {len(ident[:s].encode("utf-8")) for s in starts}


def canonical_identifier_cuts(ident: str, cuts: Iterable[int]) -> Set[int]:
    """Map every cut inside or at either edge of a delimiter run to the run's end; drop cuts at
    0/len and next to a leading or trailing run (offsets are bytes; delimiters are ASCII)."""
    n = len(ident.encode("utf-8"))
    runs = [(len(ident[: m.start()].encode("utf-8")), len(ident[: m.end()].encode("utf-8")))
            for m in re.finditer(r"[_\-]+", ident)]
    out: Set[int] = set()
    for c in cuts:
        for start, end in runs:
            if start <= c <= end:
                c = 0 if start == 0 else end
                break
        if 0 < c < n:
            out.add(c)
    return out


def mine_identifiers(code_texts: Sequence[str], max_items: int = 2000) -> List[str]:
    """The ``max_items`` most frequent identifiers with at least one gold boundary (ties: alphabetical)."""
    counts = Counter(m.group() for text in code_texts for m in IDENTIFIER_RE.finditer(text))
    multi = [ident for ident in counts if extract_identifier_cuts(ident) and len(ident) <= 64]
    multi.sort(key=lambda s: (-counts[s], s))
    return multi[:max_items]


def token_cuts(tokenizer: Any, token_ids: Sequence[int]) -> Set[int]:
    """Byte offsets between consecutive tokens, from each token's raw bytes."""
    if not hasattr(tokenizer, "id_to_bytes"):
        raise TypeError(f"{type(tokenizer).__name__} has no id_to_bytes; byte offsets are unavailable")
    cuts: Set[int] = set()
    offset = 0
    for tid in token_ids[:-1]:
        b = tokenizer.id_to_bytes(tid)
        if b is None:
            raise ValueError(f"id_to_bytes({tid}) returned None")
        offset += len(b)
        cuts.add(offset)
    return cuts


class IntrinsicEvaluator:
    """Evaluates intrinsic properties: Rényi efficiency, morphological F1, utilization, throughput."""

    def compute_renyi_from_counts(self, counts: List[int], alpha: float = 1.0) -> float:
        """H_alpha(p) / log2|V| with |V| = len(counts). No clamping: 0 <= E <= 1 and E is
        non-increasing in alpha mathematically, so a violation beyond float noise is a bug."""
        total = sum(counts)
        num_items = len(counts)
        if num_items <= 1 or total == 0:
            return 0.0

        probs = [c / total for c in counts if c > 0]
        log_v = math.log2(num_items)
        if abs(alpha - 1.0) < 1e-6:
            return -sum(p * math.log2(p) for p in probs) / log_v
        h_alpha = math.log2(sum(p ** alpha for p in probs)) / (1.0 - alpha)
        return h_alpha / log_v

    def _counts(self, tokenizer: Any, text_sample: str) -> List[int]:
        counts = [0] * tokenizer.vocab_size
        for tid in tokenizer.encode(text_sample):
            counts[tid] += 1
        return counts

    def compute_renyi_efficiency(self, tokenizer: Any, text_sample: str) -> Dict[str, float]:
        """Rényi efficiency of the unigram token distribution on ``text_sample``; |V| is the
        tokenizer's actual vocabulary size."""
        counts = self._counts(tokenizer, text_sample)
        return {
            "e_1_0": self.compute_renyi_from_counts(counts, alpha=1.0),
            "e_2_0": self.compute_renyi_from_counts(counts, alpha=2.0),
            "e_2_5": self.compute_renyi_from_counts(counts, alpha=2.5),
        }

    def compute_utilization_metrics(self, counts: List[int]) -> Dict[str, float]:
        total_slots = len(counts)
        active = sum(1 for c in counts if c > 0)
        dead = total_slots - active
        active_frac = active / max(1, total_slots)

        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        s = sum(sorted_counts)
        if s == 0 or n == 0:
            gini = 0.0
        else:
            gini = sum((2 * (i + 1) - n - 1) * val for i, val in enumerate(sorted_counts)) / (n * s)

        return {
            "active_fraction": active_frac,
            "dead_tokens": dead,
            "gini_coefficient": gini,
        }

    def compute_vocab_utilization(self, tokenizer: Any, text_sample: str) -> Dict[str, float]:
        return self.compute_utilization_metrics(self._counts(tokenizer, text_sample))

    def evaluate_cut_f1(self, pred_cuts: Set[int], gold_cuts: Set[int]) -> Dict[str, float]:
        if not pred_cuts and not gold_cuts:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        tp = len(pred_cuts & gold_cuts)
        p = tp / len(pred_cuts) if pred_cuts else 0.0
        r = tp / len(gold_cuts) if gold_cuts else 0.0
        f1 = (2.0 * p * r) / (p + r) if (p + r) > 0.0 else 0.0
        return {"precision": p, "recall": r, "f1": f1}

    def _mean_f1(self, pairs: Iterable[tuple]) -> Dict[str, float]:
        ps: List[float] = []
        rs: List[float] = []
        fs: List[float] = []
        for pred, gold in pairs:
            res = self.evaluate_cut_f1(pred, gold)
            ps.append(res["precision"])
            rs.append(res["recall"])
            fs.append(res["f1"])
        if not fs:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
        return {"precision": sum(ps) / len(ps), "recall": sum(rs) / len(rs), "f1": sum(fs) / len(fs), "n": len(fs)}

    def compute_morphological_f1(
        self, tokenizer: Any, gold_annotations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Macro (per-word) boundary precision/recall/F1 of the tokenizer's split of each gold word.
        Entries without byte-offset gold (``gold_cuts`` None: canonical, non-surface) are skipped."""

        def pairs():
            for item in gold_annotations:
                gold = item.get("gold_cuts")
                if gold is None:
                    if "gold_cuts" in item:
                        continue
                    gold, curr = set(), 0
                    for seg in item.get("segments", [])[:-1]:
                        curr += len(seg.encode("utf-8"))
                        gold.add(curr)
                yield token_cuts(tokenizer, tokenizer.encode(item["word"])), set(gold)

        return self._mean_f1(pairs())

    def extract_identifier_cuts(self, identifier: str) -> Set[int]:
        """Extracts ground-truth boundary cut set for a given code identifier."""
        return extract_identifier_cuts(identifier)

    def compute_identifier_f1(
        self,
        tokenizer: Any,
        identifiers: Optional[List[Union[str, Dict[str, Any]]]] = None,
    ) -> Dict[str, float]:
        """
        Boundary precision/recall/F1 of code identifiers (gold rule: ``extract_identifier_cuts``).
        Items are identifier strings, or dicts with ``identifier``/``word`` and optional
        ``gold_cuts``/``segments``.
        """
        items = identifiers if identifiers is not None else DEFAULT_CODE_IDENTIFIERS

        def pairs():
            for item in items:
                if isinstance(item, str):
                    ident, gold = item, None
                else:
                    ident = item.get("identifier") or item.get("word") or ""
                    gold = item.get("gold_cuts")
                    if gold is None and "segments" in item:
                        gold, curr = set(), 0
                        for s in item["segments"][:-1]:
                            curr += len(s.encode("utf-8"))
                            gold.add(curr)
                if gold is None:
                    gold = extract_identifier_cuts(ident)
                pred = token_cuts(tokenizer, tokenizer.encode(ident))
                yield canonical_identifier_cuts(ident, pred), canonical_identifier_cuts(ident, gold)

        return self._mean_f1(pairs())

    def compute_number_fertility(self, tokenizer: Any, number_str: str) -> float:
        return float(len(tokenizer.encode(number_str)))

    def compute_structural_diagnostics(
        self,
        tokenizer: Any,
        number_str: str = "1234567890",
        identifiers: Optional[List[Union[str, Dict[str, Any]]]] = None,
    ) -> Dict[str, float]:
        """Number fertility (tokens for ``number_str``) and identifier boundary P/R/F1."""
        ident_res = self.compute_identifier_f1(tokenizer, identifiers)
        return {
            "number_fertility": self.compute_number_fertility(tokenizer, number_str),
            "identifier_f1": ident_res["f1"],
            "identifier_precision": ident_res["precision"],
            "identifier_recall": ident_res["recall"],
            "identifiers": ident_res["n"],
        }

    def benchmark_throughput(
        self,
        tokenizer: Any,
        text_sample: Union[str, Sequence[str]],
        num_warmup: int = 1,
        repeats: int = 5,
    ) -> Dict[str, float]:
        """
        Encode/decode throughput, one ``encode``/``decode`` call per document as in practice.
        Reports the median over ``repeats`` timed passes (after ``num_warmup`` untimed ones) of
        total bytes / total time. No floor is applied.
        """
        docs = [text_sample] if isinstance(text_sample, str) else list(text_sample)
        raw_bytes = sum(len(d.encode("utf-8")) for d in docs)
        mb = raw_bytes / (1024 * 1024)
        enc_times: List[float] = []
        dec_times: List[float] = []
        for i in range(num_warmup + repeats):
            t0 = time.perf_counter()
            ids = [tokenizer.encode(d) for d in docs]
            t1 = time.perf_counter()
            for x in ids:
                tokenizer.decode(x)
            t2 = time.perf_counter()
            if i >= num_warmup:
                enc_times.append(t1 - t0)
                dec_times.append(t2 - t1)
        enc = statistics.median(enc_times)
        dec = statistics.median(dec_times)
        return {
            "encode_mb_per_sec": mb / enc if enc > 0 else math.inf,
            "decode_mb_per_sec": mb / dec if dec > 0 else math.inf,
            "throughput_bytes": raw_bytes,
            "throughput_docs": len(docs),
            "throughput_repeats": repeats,
        }

    def evaluate_suite(
        self,
        models: Dict[str, Any],
        text_sample: Union[str, Sequence[str]],
        gold_annotations: List[Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
        identifiers: Optional[List[Union[str, Dict[str, Any]]]] = None,
        meta: Optional[Dict[str, Any]] = None,
        repeats: int = 5,
        throughput_docs: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs the suite on ``text_sample`` (one string or a list of documents; documents are
        encoded one by one) and writes intrinsic_leaderboard.{json,csv,md} when ``output_dir`` is
        given. Morph F1 is reported overall and per gold ``lang``. Throughput is timed on
        ``throughput_docs`` (default: the evaluation documents).
        """
        docs = [text_sample] if isinstance(text_sample, str) else list(text_sample)
        doc_bytes = sum(len(d.encode("utf-8")) for d in docs)
        words = [w for d in docs for w in d.split()]
        num_words = max(1, len(words))
        by_lang: Dict[str, List[Dict[str, Any]]] = {}
        for g in gold_annotations:
            by_lang.setdefault(g.get("lang") or "all", []).append(g)

        results: Dict[str, Dict[str, Any]] = {}
        for name, tok in models.items():
            counts = [0] * tok.vocab_size
            n_toks = 0
            for d in docs:
                for tid in tok.encode(d):
                    counts[tid] += 1
                    n_toks += 1
            morph_by_lang = {lang: self.compute_morphological_f1(tok, items) for lang, items in sorted(by_lang.items())}
            n_morph = sum(m["n"] for m in morph_by_lang.values())
            morph = {k: sum(m[k] * m["n"] for m in morph_by_lang.values()) / max(1, n_morph)
                     for k in ("precision", "recall", "f1")}
            struct = self.compute_structural_diagnostics(tok, identifiers=identifiers)
            speed = self.benchmark_throughput(tok, throughput_docs or docs, repeats=repeats)
            util = self.compute_utilization_metrics(counts)
            results[name] = {
                "vocab_size": tok.vocab_size,
                "bytes_per_token": doc_bytes / max(1, n_toks),
                "fertility": sum(len(tok.encode(w)) for w in words) / num_words,
                "renyi_e_1_0": self.compute_renyi_from_counts(counts, 1.0),
                "renyi_e_2_0": self.compute_renyi_from_counts(counts, 2.0),
                "renyi_e_2_5": self.compute_renyi_from_counts(counts, 2.5),
                "active_fraction": util["active_fraction"],
                "dead_tokens": util["dead_tokens"],
                "gini_coefficient": util["gini_coefficient"],
                "morph_precision": morph["precision"],
                "morph_recall": morph["recall"],
                "morph_f1": morph["f1"],
                "morph_words": n_morph,
                "morph_f1_by_lang": {lang: m["f1"] for lang, m in morph_by_lang.items()},
                "identifier_precision": struct["identifier_precision"],
                "identifier_recall": struct["identifier_recall"],
                "identifier_f1": struct["identifier_f1"],
                "identifiers": struct["identifiers"],
                "number_fertility": struct["number_fertility"],
                **speed,
            }

        if output_dir is not None:
            from tokenisers.benchmarks.report import render_intrinsic

            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            doc = {"meta": meta or {}, "models": results}
            (out_dir / "intrinsic_leaderboard.json").write_text(
                json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            fieldnames = [
                "tokenizer", "vocab_size", "bytes_per_token", "fertility", "renyi_e_1_0", "renyi_e_2_0",
                "renyi_e_2_5", "active_fraction", "morph_f1", "identifier_f1", "number_fertility",
                "encode_mb_per_sec", "decode_mb_per_sec",
            ]
            with open(out_dir / "intrinsic_leaderboard.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                for m_name, vals in results.items():
                    writer.writerow([m_name] + [vals[k] for k in fieldnames[1:]])
            (out_dir / "intrinsic_leaderboard.md").write_text(render_intrinsic(doc) + "\n", encoding="utf-8")

        return results


__all__ = [
    "IntrinsicEvaluator",
    "extract_identifier_cuts",
    "canonical_identifier_cuts",
    "mine_identifiers",
    "token_cuts",
    "DEFAULT_CODE_IDENTIFIERS",
]
