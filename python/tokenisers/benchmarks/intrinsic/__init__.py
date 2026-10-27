"""
Extended Intrinsic Tokenizer Evaluation Suite.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional, Set, Union

DEFAULT_CODE_IDENTIFIERS: List[str] = [
    "getUserProfile",
    "get_user_profile",
    "calculate_loss_v2",
    "HttpRequestHandler",
    "max_retries_count",
]


def extract_identifier_cuts(ident: str) -> Set[int]:
    """
    Extracts ground-truth word boundary cuts for a programming code identifier.

    Handles camelCase, PascalCase, snake_case, screaming snake case, and version numbers.
    Returns cut positions as byte/character offsets from the start of the identifier.
    """
    raw_parts = re.split(r"(_+|-+)", ident)
    segments: List[str] = []
    for p in raw_parts:
        if not p:
            continue
        if p.startswith("_") or p.startswith("-"):
            segments.append(p)
        else:
            sub = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+[0-9]*|[0-9]+", p)
            if sub and "".join(sub) == p:
                segments.extend(sub)
            else:
                segments.append(p)

    cuts: Set[int] = set()
    offset = 0
    for seg in segments[:-1]:
        offset += len(seg.encode("utf-8"))
        cuts.add(offset)
    return cuts



class IntrinsicEvaluator:
    """Evaluates intrinsic properties: Rényi efficiency, morphological F1, utilization, throughput."""

    def compute_renyi_from_counts(self, counts: List[int], alpha: float = 1.0) -> float:
        total = sum(counts)
        num_items = len(counts)
        if num_items <= 1 or total == 0:
            return 0.0

        probs = [c / total for c in counts if c > 0]
        log_v = math.log2(num_items)
        if log_v <= 0.0:
            return 0.0

        if abs(alpha - 1.0) < 1e-6:
            h = -sum(p * math.log2(p) for p in probs)
            return max(0.0, min(1.0, h / log_v))

        sum_p_alpha = sum(p ** alpha for p in probs)
        if sum_p_alpha <= 0.0:
            return 0.0
        h_alpha = (1.0 / (1.0 - alpha)) * math.log2(sum_p_alpha)
        return max(0.0, min(1.0, h_alpha / log_v))

    def compute_renyi_efficiency(self, tokenizer: Any, text_sample: str) -> Dict[str, float]:
        tokens = tokenizer.encode(text_sample)
        vocab_size = getattr(tokenizer, "vocab_size", 256)
        counts = [0] * vocab_size
        for tid in tokens:
            if tid < vocab_size:
                counts[tid] += 1

        e1 = self.compute_renyi_from_counts(counts, alpha=1.0)
        e2 = self.compute_renyi_from_counts(counts, alpha=2.0)
        e25 = self.compute_renyi_from_counts(counts, alpha=2.5)

        # Enforce mathematical monotonic inequality: e25 <= e2 <= e1
        e2 = min(e1, e2)
        e25 = min(e2, e25)

        return {
            "e_1_0": e1,
            "e_2_0": e2,
            "e_2_5": e25,
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
            cum = sum((2 * (i + 1) - n - 1) * val for i, val in enumerate(sorted_counts))
            gini = max(0.0, min(1.0, cum / (n * s)))

        return {
            "active_fraction": active_frac,
            "dead_tokens": dead,
            "gini_coefficient": gini,
        }

    def compute_vocab_utilization(self, tokenizer: Any, text_sample: str) -> Dict[str, float]:
        tokens = tokenizer.encode(text_sample)
        vocab_size = getattr(tokenizer, "vocab_size", 256)
        counts = [0] * vocab_size
        for tid in tokens:
            if tid < vocab_size:
                counts[tid] += 1
        return self.compute_utilization_metrics(counts)

    def evaluate_cut_f1(self, pred_cuts: Set[int], gold_cuts: Set[int]) -> Dict[str, float]:
        if not pred_cuts and not gold_cuts:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        tp = len(pred_cuts & gold_cuts)
        p = tp / len(pred_cuts) if pred_cuts else 0.0
        r = tp / len(gold_cuts) if gold_cuts else 0.0
        f1 = (2.0 * p * r) / (p + r) if (p + r) > 0.0 else 0.0
        return {"precision": p, "recall": r, "f1": f1}

    def compute_morphological_f1(
        self, tokenizer: Any, gold_annotations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        if not gold_annotations:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precisions = []
        recalls = []
        f1s = []

        for item in gold_annotations:
            word = item["word"]
            if "gold_cuts" in item:
                gold_cuts = set(item["gold_cuts"])
            else:
                gold_cuts = set()
                curr = 0
                for seg in item.get("segments", [])[:-1]:
                    curr += len(seg.encode("utf-8"))
                    gold_cuts.add(curr)

            tokens = tokenizer.encode(word)
            pred_cuts: Set[int] = set()
            offset = 0

            for tid in tokens[:-1]:
                tok_str = getattr(tokenizer, "id_to_token", lambda x: str(x))(tid)
                tok_bytes = getattr(tokenizer, "id_to_bytes", lambda x: None)(tid)
                if tok_bytes is not None:
                    offset += len(tok_bytes)
                elif tok_str is not None:
                    offset += len(tok_str.encode("utf-8"))
                else:
                    offset += 1
                pred_cuts.add(offset)

            res = self.evaluate_cut_f1(pred_cuts, gold_cuts)
            precisions.append(res["precision"])
            recalls.append(res["recall"])
            f1s.append(res["f1"])

        return {
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "f1": sum(f1s) / len(f1s),
        }

    def extract_identifier_cuts(self, identifier: str) -> Set[int]:
        """Extracts ground-truth boundary cut set for a given code identifier."""
        return extract_identifier_cuts(identifier)

    def compute_identifier_f1(
        self,
        tokenizer: Any,
        identifiers: Optional[List[Union[str, Dict[str, Any]]]] = None,
    ) -> Dict[str, float]:
        """
        Computes genuine subword boundary Precision, Recall, and F1 for programming identifiers.
        """
        items = identifiers if identifiers is not None else DEFAULT_CODE_IDENTIFIERS
        if not items:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []

        for item in items:
            if isinstance(item, str):
                ident_str = item
                gold_cuts = extract_identifier_cuts(ident_str)
            elif isinstance(item, dict):
                ident_str = item.get("identifier") or item.get("word") or ""
                if "gold_cuts" in item:
                    gold_cuts = set(item["gold_cuts"])
                elif "segments" in item:
                    gold_cuts = set()
                    curr = 0
                    for s in item["segments"][:-1]:
                        curr += len(s.encode("utf-8"))
                        gold_cuts.add(curr)
                else:
                    gold_cuts = extract_identifier_cuts(ident_str)
            else:
                continue

            tokens = tokenizer.encode(ident_str)
            pred_cuts: Set[int] = set()

            if hasattr(tokenizer, "decode") and len(tokens) > 1:
                for i in range(len(tokens) - 1):
                    try:
                        prefix = tokenizer.decode(tokens[: i + 1])
                        pred_cuts.add(len(prefix.encode("utf-8")))
                    except Exception:
                        break

            if not pred_cuts and len(tokens) > 1:
                offset = 0
                for tid in tokens[:-1]:
                    tok_bytes = None
                    if hasattr(tokenizer, "id_to_bytes"):
                        tok_bytes = tokenizer.id_to_bytes(tid)
                    elif hasattr(tokenizer, "id_to_token"):
                        tok_str = tokenizer.id_to_token(tid)
                        if tok_str is not None:
                            tok_bytes = tok_str.encode("utf-8")
                    if tok_bytes is not None:
                        offset += len(tok_bytes)
                    else:
                        offset += 1
                    pred_cuts.add(offset)

            res = self.evaluate_cut_f1(pred_cuts, gold_cuts)
            precisions.append(res["precision"])
            recalls.append(res["recall"])
            f1s.append(res["f1"])

        p = sum(precisions) / len(precisions) if precisions else 0.0
        r = sum(recalls) / len(recalls) if recalls else 0.0
        f1 = sum(f1s) / len(f1s) if f1s else 0.0

        return {"precision": p, "recall": r, "f1": f1}

    def compute_number_fertility(self, tokenizer: Any, number_str: str) -> float:
        tokens = tokenizer.encode(number_str)
        return float(len(tokens))

    def compute_structural_diagnostics(
        self,
        tokenizer: Any,
        number_str: str = "1234567890",
        identifiers: Optional[List[Union[str, Dict[str, Any]]]] = None,
    ) -> Dict[str, float]:
        """
        Computes genuine structural diagnostics:
        1. Number fertility: number of tokens produced for numeric strings.
        2. Identifier boundary metrics: genuine Precision, Recall, and F1 against camelCase/snake_case cuts.
        """
        num_fert = self.compute_number_fertility(tokenizer, number_str)
        ident_res = self.compute_identifier_f1(tokenizer, identifiers)
        return {
            "number_fertility": num_fert,
            "identifier_f1": ident_res["f1"],
            "identifier_precision": ident_res["precision"],
            "identifier_recall": ident_res["recall"],
        }

    def benchmark_throughput(
        self, tokenizer: Any, text_sample: str, num_warmup: int = 2
    ) -> Dict[str, float]:
        raw_bytes = len(text_sample.encode("utf-8"))

        for _ in range(num_warmup):
            ids = tokenizer.encode(text_sample)
            _ = tokenizer.decode(ids)

        # Measure encode
        t0 = time.perf_counter()
        ids = tokenizer.encode(text_sample)
        t_enc = max(time.perf_counter() - t0, 1e-6)

        # Measure decode
        t0 = time.perf_counter()
        _ = tokenizer.decode(ids)
        t_dec = max(time.perf_counter() - t0, 1e-6)

        mb = raw_bytes / (1024 * 1024)
        return {
            "encode_mb_per_sec": max(0.1, mb / t_enc),
            "decode_mb_per_sec": max(0.1, mb / t_dec),
        }


__all__ = ["IntrinsicEvaluator", "extract_identifier_cuts", "DEFAULT_CODE_IDENTIFIERS"]

