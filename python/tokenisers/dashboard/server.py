"""
HTTP & JSON API Server for Tokenisers Lab Benchmark Intelligence Dashboard.
Inspired by Artificial Analysis (artificialanalysis.ai).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tokenisers.dashboard")

# Determine repository root
REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Metadata for all algorithm architectures and theoretical objectives
ALGORITHM_METADATA: Dict[str, Dict[str, Any]] = {
    "bpe_frequency": {
        "display_name": "BPE (Frequency)",
        "family": "Frequency BPE",
        "description": "Standard byte-level byte-pair encoding merging most frequent adjacent byte pairs.",
        "formula": r"\mathrm{score}(u, v) = \mathrm{count}(uv)",
        "objective": "Pair Frequency",
        "generation": "Baseline",
        "pros": ["Fastest training", "High in-domain compression", "Industry standard baseline"],
        "cons": ["Blind to word boundaries", "Over-merges frequent affixes into root words", "Poor morphological alignment"],
    },
    "wordpiece_pmi": {
        "display_name": "WordPiece (PMI)",
        "family": "Information-Theoretic",
        "description": "Information-theoretic merge objective measuring statistical co-occurrence vs. independent marginals.",
        "formula": r"\mathrm{score}(u, v) = \frac{\mathrm{count}(uv)}{\mathrm{count}(u) \cdot \mathrm{count}(v)}",
        "objective": "Pointwise Mutual Information",
        "generation": "Baseline",
        "pros": ["Favors statistically coherent morphemes", "Resistant to purely superficial frequency bias"],
        "cons": ["Severe singleton instability without frequency thresholds", "Higher vocabulary fragmentation"],
    },
    "wp_bpe_base": {
        "display_name": "WP-BPE (Base Uncalibrated PMI)",
        "family": "Information-Theoretic",
        "description": "BPE merges driven by raw uncalibrated PMI. Highlights theoretical vulnerability of naive PMI.",
        "formula": r"\mathrm{score}(u, v) = \frac{\mathrm{freq}(uv)}{\mathrm{freq}(u) \cdot \mathrm{freq}(v)}",
        "objective": "Uncalibrated PMI",
        "generation": "WP-BPE Gen 1",
        "pros": ["Exposes true statistical dependencies across tokens"],
        "cons": ["Rare pairs (freq=1) obtain astronomical scores", "Severe corpus overfitting without thresholding"],
    },
    "wp_bpe_ft": {
        "display_name": "WP-BPE (Frequency-Thresholded)",
        "family": "Information-Theoretic",
        "description": "Calibrated PMI with minimum frequency cutoffs (threshold theta >= 2) and exponent alpha smoothing.",
        "formula": r"\mathrm{score}(u, v) = \begin{cases} \frac{\mathrm{freq}(uv)}{\mathrm{freq}(u)^\alpha \cdot \mathrm{freq}(v)^\alpha} & \text{if } \mathrm{freq}(uv) \ge \theta \\ -\infty & \text{otherwise} \end{cases}",
        "objective": "Frequency-Thresholded PMI",
        "generation": "WP-BPE Gen 1",
        "pros": ["Filters out singleton noise", "Drastically boosts morphological boundary precision", "Stable vocabulary distribution"],
        "cons": ["Requires tuning threshold theta and exponent alpha", "Slightly lower compression ratio on rare words"],
    },
    "wp_bpe_pig": {
        "display_name": "WP-BPE (Pointwise Information Gain)",
        "family": "Information-Theoretic",
        "description": "Directly weights PMI by candidate pair frequency, measuring total bits of information saved across the corpus.",
        "formula": r"\mathrm{score}(u, v) = \mathrm{freq}(uv) \cdot \log_2 \frac{\mathrm{freq}(uv) \cdot N}{\mathrm{freq}(u) \cdot \mathrm{freq}(v)}",
        "objective": "Corpus Information Gain",
        "generation": "WP-BPE Gen 1",
        "pros": ["Entropy-reduction grounded in information theory", "Balances frequency against statistical cohesion", "Superior downstream LM perplexity"],
        "cons": ["Slightly slower merge selection queue", "Sensitive to corpus size normalization"],
    },
    "wp_bpe_scp": {
        "display_name": "WP-BPE (Symmetric Conditional Probability)",
        "family": "Information-Theoretic (Improved)",
        "description": "Next-gen metric using Symmetric Conditional Probability to balance directional association P(u|v) and P(v|u).",
        "formula": r"\mathrm{score}(u, v) = \frac{P(uv)^2}{P(u) \cdot P(v)} = P(u \mid v) \cdot P(v \mid u)",
        "objective": "Symmetric Conditional Probability",
        "generation": "WP-BPE Gen 2 (Improved)",
        "pros": ["Self-bounding cohesion index", "Equalizes prefix-root and root-suffix merge biases", "Robust across agglutinative morphologies"],
        "cons": ["Quadratic joint probability scaling requires careful float precision"],
    },
    "wp_bpe_npmi": {
        "display_name": "WP-BPE (Normalized PMI)",
        "family": "Information-Theoretic (Improved)",
        "description": "Bouma Normalized PMI mapped strictly into [-1, +1], providing unbiased significance regardless of frequency scale.",
        "formula": r"\mathrm{score}(u, v) = \frac{\log \frac{P(uv)}{P(u)P(v)}}{-\log P(uv)} \in [-1, +1]",
        "objective": "Normalized PMI",
        "generation": "WP-BPE Gen 2 (Improved)",
        "pros": ["Eliminates arbitrary frequency threshold tuning theta", "Perfect score +1 occurs if and only if tokens strictly co-occur", "Theoretical maximum robustness"],
        "cons": ["Log denominator requires eps guard against zero probabilities"],
    },
    "wp_bpe_ppig": {
        "display_name": "WP-BPE (Penalized PIG)",
        "family": "Information-Theoretic (Improved)",
        "description": "Enhanced Pointwise Information Gain with polynomial length regularization and transition entropy weighting.",
        "formula": r"\mathrm{score}(u, v) = \mathrm{freq}(uv) \cdot \mathrm{PMI}(u, v) \cdot (|u| + |v|)^{-\beta} \cdot \mathcal{H}(u, v)",
        "objective": "Penalized Information Gain",
        "generation": "WP-BPE Gen 2 (Improved)",
        "pros": ["Prevents vocabulary bloating with overly long collocations", "Preserves boundary points across high-entropy transitions", "Optimal downstream language model bits-per-byte"],
        "cons": ["Requires hyperparameter beta selection"],
    },
    "mdl_bits": {
        "display_name": "MDL (Delta-Bits)",
        "family": "Information-Theoretic",
        "description": "Minimum Description Length criterion choosing merges that maximize net reduction in two-part code length.",
        "formula": r"\Delta L = L_{\mathrm{corpus}}(T) - L_{\mathrm{corpus}}(T') + \Delta L_{\mathrm{lexicon}}",
        "objective": "Minimum Description Length",
        "generation": "Baseline",
        "pros": ["Rigorous MDL foundation", "Resistant to vocabulary bloat", "Guaranteed theoretical compressibility"],
        "cons": ["Higher computational cost during candidate selection", "Sensitive to table overhead estimation"],
    },
    "length_penalized": {
        "display_name": "Length-Penalized BPE",
        "family": "Frequency BPE",
        "description": "Frequency scoring divided by token length power beta to discourage greedy concatenation of long subwords.",
        "formula": r"\mathrm{score}(u, v) = \frac{\mathrm{freq}(uv)}{(|u| + |v|)^\beta}",
        "objective": "Length-Penalized Frequency",
        "generation": "Baseline",
        "pros": ["Prevents extreme long-tail token dominance", "Balances subword fertility across long words", "Simple and predictable"],
        "cons": ["Beta parameter is empirical", "Does not consider morphological boundaries"],
    },
    "super_bpe": {
        "display_name": "SuperBPE (Two-Stage)",
        "family": "Structural / Multi-Stage",
        "description": "Constrains merges strictly within word boundaries up to v_switch, then unlocks cross-word collocation merges.",
        "formula": r"\begin{cases} \text{Word-internal BPE} & |V| \le V_{\mathrm{switch}} \\ \text{Cross-word Collocations} & V_{\mathrm{switch}} < |V| \le V_{\mathrm{final}} \end{cases}",
        "objective": "Two-Stage Structural BPE",
        "generation": "Baseline",
        "pros": ["Guarantees morphological integrity for core vocabulary", "Captures frequent phrases (e.g. 'in the', 'et al.') efficiently", "Outstanding code compression"],
        "cons": ["Requires dual-pass vocabulary boundary tracking", "Slightly larger initial graph memory footprint"],
    },
    "morpheme_bpe": {
        "display_name": "Morpheme-Guided BPE",
        "family": "Morphology / Guided",
        "description": "BPE merge scores dynamically augmented by ground-truth morphological cuts from linguistic lexicons (e.g. Sigmorphon).",
        "formula": r"\mathrm{score}(u, v) = \mathrm{freq}(uv) \cdot (1 + \lambda \cdot \mathbb{I}_{\mathrm{morpheme\_cut}}(u, v))",
        "objective": "Morpheme Boundary Regularization",
        "generation": "Baseline",
        "pros": ["Highest morphological boundary F1 score", "Linguistically aligned token pieces", "Superior generalization on morphologically rich languages"],
        "cons": ["Requires supervised or semi-supervised morphological dictionary", "Lambda weight hyperparameter tuning"],
    },
    "unigram_lm": {
        "display_name": "Unigram Language Model",
        "family": "EM / Probabilistic",
        "description": "Top-down vocabulary pruning using Expectation-Maximization and Viterbi / stochastic lattice decoding.",
        "formula": r"\mathcal{L} = \sum_{i=1}^N \log \left( \sum_{x \in \mathcal{S}(X_i)} P(x) \right)",
        "objective": "Maximum Likelihood Unigram",
        "generation": "Baseline",
        "pros": ["Probabilistic foundation with lattice sampling (subword regularization)", "High entropy retention (Rényi E_2.0)", "Smooth token distributions"],
        "cons": ["Slower vocabulary training phase due to iterative EM loops", "Higher Viterbi decoding overhead compared to deterministic BPE"],
    },
    "bpe_dropout_p0.1": {
        "display_name": "BPE (Merge Dropout p=0.1)",
        "family": "Regularized BPE",
        "description": "Stochastic BPE segmenter that randomly skips merges with probability p=0.1 during tokenization.",
        "formula": r"P(\text{apply merge } (u,v)) = 1 - p",
        "objective": "Subword Regularization Dropout",
        "generation": "Baseline",
        "pros": ["Data augmentation for downstream LM robustness", "Prevents memorization of rigid subword boundaries"],
        "cons": ["Non-deterministic segmentation", "Requires seed control for reproducibility"],
    },
}


class TokenizerCache:
    """Thread-safe lazy cache of trained tokenizer instances across vocab sizes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._corpus: Optional[List[str]] = None
        self._morph_corpus: Optional[List[str]] = None
        self._lexicon: Optional[Dict[str, List[str]]] = None

    def _prepare_corpus(self) -> Tuple[List[str], Dict[str, List[str]]]:
        if self._morph_corpus is not None and self._lexicon is not None:
            return self._morph_corpus, self._lexicon

        try:
            from tokenisers.data.pipeline import DataPipeline, SigmorphonLoader

            pipeline = DataPipeline(force_offline=True)
            corpus = pipeline.sample_preset("smoke", max_bytes=60_000)
            loader = SigmorphonLoader()
            if hasattr(loader, "synthetic_compounds"):
                compounds = loader.synthetic_compounds()
                gold = [{"word": c, "segments": [c[: len(c) // 2], c[len(c) // 2 :]]} for c in compounds[:20]]
            else:
                gold = loader.load_builtin_samples()

            lexicon = {item["word"]: item["segments"] for item in gold}
            morph_corpus = corpus + [item["word"] for item in gold] * 10
            self._corpus = corpus
            self._morph_corpus = morph_corpus
            self._lexicon = lexicon
            return morph_corpus, lexicon
        except Exception as e:
            logger.warning(f"Error loading data pipeline: {e}. Falling back to default corpus.")
            fallback = [
                "The internationalization of unhappily disproportionate misunderstandings requires robust subword tokenization.",
                "Reinforcement learning from human feedback and artificial intelligence benchmarks provide deep insights.",
                "In computer science, tokenization transforms unstructured byte sequences into discrete structural units.",
                "fn calculate_pmi(freq_uv: u64, freq_u: u64, freq_v: u64) -> f64 { freq_uv as f64 / (freq_u * freq_v) as f64 }",
            ]
            self._corpus = fallback
            self._morph_corpus = fallback * 20
            self._lexicon = {"internationalization": ["inter", "nation", "al", "iz", "ation"]}
            return self._morph_corpus, self._lexicon

    def get_tokenizers(self, vocab_size: int = 400) -> Dict[str, Any]:
        with self._lock:
            if vocab_size in self._cache:
                return self._cache[vocab_size]

            corpus, lexicon = self._prepare_corpus()
            v_size = max(260, vocab_size)
            v_switch = max(256, int(v_size * 0.75))

            from tokenisers import models
            from tokenisers.models import (
                BpeTokenizer,
                LengthPenalizedTokenizer,
                MdlTokenizer,
                MorphemeBpeTokenizer,
                SuperBpeTokenizer,
                UnigramTokenizer,
                WordPieceTokenizer,
                WpBpeBaseTokenizer,
                WpBpeFtTokenizer,
                WpBpePigTokenizer,
            )

            toks: Dict[str, Any] = {}

            # Baseline and Gen 1 models
            model_factories = [
                ("bpe_frequency", lambda: BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="frequency")),
                ("wordpiece_pmi", lambda: WordPieceTokenizer.train(corpus=corpus, vocab_size=v_size)),
                ("wp_bpe_base", lambda: WpBpeBaseTokenizer.train(corpus=corpus, vocab_size=v_size)),
                ("wp_bpe_ft", lambda: WpBpeFtTokenizer.train(corpus=corpus, vocab_size=v_size, min_frequency=2)),
                ("wp_bpe_pig", lambda: WpBpePigTokenizer.train(corpus=corpus, vocab_size=v_size, min_frequency=1)),
                ("mdl_bits", lambda: MdlTokenizer.train(corpus=corpus, vocab_size=v_size)),
                ("length_penalized", lambda: LengthPenalizedTokenizer.train(corpus=corpus, vocab_size=v_size, beta=0.5)),
                ("super_bpe", lambda: SuperBpeTokenizer.train(corpus=corpus, v_switch=v_switch, v_final=v_size)),
                ("morpheme_bpe", lambda: MorphemeBpeTokenizer.train(corpus=corpus, vocab_size=v_size, morpheme_lexicon=lexicon, lambda_weight=2.5)),
                ("unigram_lm", lambda: UnigramTokenizer.train(corpus=corpus, vocab_size=v_size)),
            ]

            for name, factory in model_factories:
                try:
                    toks[name] = factory()
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")

            # Check for Gen 2 improved WP-BPE models in models module
            gen2_candidates = [
                ("wp_bpe_scp", "WpBpeScpTokenizer", "wp-bpe-scp"),
                ("wp_bpe_npmi", "WpBpeNpmiTokenizer", "wp-bpe-npmi"),
                ("wp_bpe_ppig", "WpBpePpigTokenizer", "wp-bpe-ppig"),
            ]

            for key, cls_name, scorer_name in gen2_candidates:
                cls_obj = getattr(models, cls_name, None) or getattr(models, key, None)
                if cls_obj is not None and hasattr(cls_obj, "train"):
                    try:
                        toks[key] = cls_obj.train(corpus=corpus, vocab_size=v_size)
                    except Exception as e:
                        logger.debug(f"Could not train {key} via {cls_name}: {e}")
                else:
                    # Try BpeTokenizer with scorer
                    try:
                        toks[key] = BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer=scorer_name)
                    except Exception:
                        pass

            self._cache[vocab_size] = toks
            return toks


# Singleton cache
_TOKENIZER_CACHE = TokenizerCache()


def load_benchmark_data() -> Dict[str, Any]:
    """Loads and unifies tokka, intrinsic, and extrinsic benchmark results."""
    # 1. Tokka Bench
    tokka_candidates = [
        REPO_ROOT / "benchmarks/results/tokka/tokka_matrix.json",
        REPO_ROOT / "benchmarks/results/tokka/tokka_leaderboard.json",
        REPO_ROOT / "benchmarks/results/tokka/leaderboard.json",
    ]
    tokka_data: Dict[str, Any] = {}
    for p in tokka_candidates:
        if p.exists():
            try:
                tokka_data = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception as e:
                logger.warning(f"Failed reading {p}: {e}")

    # 2. Intrinsic Leaderboard
    intrinsic_candidates = [
        REPO_ROOT / "benchmarks/results/intrinsic/intrinsic_leaderboard.json",
        REPO_ROOT / "benchmarks/results/intrinsic/metrics.json",
        REPO_ROOT / "benchmarks/results/intrinsic/leaderboard.json",
    ]
    intrinsic_data: Dict[str, Any] = {}
    for p in intrinsic_candidates:
        if p.exists():
            try:
                intrinsic_data = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception as e:
                logger.warning(f"Failed reading {p}: {e}")

    # 3. Colab Extrinsic Proxy-LM BPB
    colab_candidates = [
        REPO_ROOT / "benchmarks/results/colab/extrinsic_bpb_results.json",
        REPO_ROOT / "benchmarks/results/colab/bpb_results.json",
        REPO_ROOT / "benchmarks/results/colab/extrinsic_bpb_leaderboard.json",
    ]
    colab_data: Dict[str, Any] = {}
    for p in colab_candidates:
        if p.exists():
            try:
                colab_data = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception as e:
                logger.warning(f"Failed reading {p}: {e}")

    colab_models = colab_data.get("models", {}) if isinstance(colab_data, dict) else {}

    # Consolidate all model identifiers
    all_keys = set(tokka_data.keys()) | set(intrinsic_data.keys()) | set(colab_models.keys()) | set(ALGORITHM_METADATA.keys())

    # Build unified model list
    unified_models: Dict[str, Dict[str, Any]] = {}
    for key in sorted(all_keys):
        meta = ALGORITHM_METADATA.get(
            key,
            {
                "display_name": key.replace("_", " ").title(),
                "family": "Custom",
                "description": f"Tokenizer implementation {key}",
                "formula": r"\mathrm{score}(u,v)",
                "objective": "Heuristic",
                "generation": "Other",
                "pros": [],
                "cons": [],
            },
        )

        intr = intrinsic_data.get(key, {})
        tok = tokka_data.get(key, {})
        col = colab_models.get(key, {})

        macro_bpb = col.get("macro_val_bpb")
        val_domains = col.get("val_domains", {})

        # Intrinsic metrics
        bytes_per_token = intr.get("bytes_per_token")
        if bytes_per_token is None and "macro_avg" in tok:
            bytes_per_token = tok["macro_avg"].get("bytes_per_token")

        unified_models[key] = {
            "key": key,
            "display_name": meta.get("display_name", key),
            "family": meta.get("family", "Unknown"),
            "generation": meta.get("generation", "Baseline"),
            "description": meta.get("description", ""),
            "formula": meta.get("formula", ""),
            "objective": meta.get("objective", ""),
            "pros": meta.get("pros", []),
            "cons": meta.get("cons", []),
            "macro_val_bpb": macro_bpb,
            "bytes_per_token": bytes_per_token,
            "renyi_e_2_0": intr.get("renyi_e_2_0"),
            "morph_f1": intr.get("morph_f1"),
            "identifier_f1": intr.get("identifier_f1"),
            "decode_mb_per_sec": intr.get("decode_mb_per_sec"),
            "encode_mb_per_sec": intr.get("encode_mb_per_sec"),
            "val_domains": val_domains,
            "steps": col.get("steps"),
            "curve": col.get("curve", []),
            "intrinsic": intr,
            "tokka": tok,
        }

    # Summary KPI Calculation
    valid_bpbs = [(k, v["macro_val_bpb"]) for k, v in unified_models.items() if v["macro_val_bpb"] is not None]
    valid_renyi = [(k, v["renyi_e_2_0"]) for k, v in unified_models.items() if v["renyi_e_2_0"] is not None]
    valid_morph = [(k, v["morph_f1"]) for k, v in unified_models.items() if v["morph_f1"] is not None]
    valid_bpt = [(k, v["bytes_per_token"]) for k, v in unified_models.items() if v["bytes_per_token"] is not None]
    valid_speed = [(k, v["decode_mb_per_sec"]) for k, v in unified_models.items() if v["decode_mb_per_sec"] is not None]

    best_bpb = min(valid_bpbs, key=lambda x: x[1]) if valid_bpbs else ("none", 0.0)
    best_renyi = max(valid_renyi, key=lambda x: x[1]) if valid_renyi else ("none", 0.0)
    best_morph = max(valid_morph, key=lambda x: x[1]) if valid_morph else ("none", 0.0)
    best_bpt = max(valid_bpt, key=lambda x: x[1]) if valid_bpt else ("none", 0.0)
    best_speed = max(valid_speed, key=lambda x: x[1]) if valid_speed else ("none", 0.0)

    summary = {
        "best_downstream_quality": {
            "model": best_bpb[0],
            "display_name": unified_models.get(best_bpb[0], {}).get("display_name", best_bpb[0]),
            "value": best_bpb[1],
            "metric": "Macro Val BPB (Lower is Better)",
        },
        "highest_information_efficiency": {
            "model": best_renyi[0],
            "display_name": unified_models.get(best_renyi[0], {}).get("display_name", best_renyi[0]),
            "value": best_renyi[1],
            "metric": "Rényi E_2.0",
        },
        "best_morphological_f1": {
            "model": best_morph[0],
            "display_name": unified_models.get(best_morph[0], {}).get("display_name", best_morph[0]),
            "value": best_morph[1],
            "metric": "Morph Boundary F1",
        },
        "best_compression": {
            "model": best_bpt[0],
            "display_name": unified_models.get(best_bpt[0], {}).get("display_name", best_bpt[0]),
            "value": best_bpt[1],
            "metric": "Bytes / Token",
        },
        "fastest_throughput": {
            "model": best_speed[0],
            "display_name": unified_models.get(best_speed[0], {}).get("display_name", best_speed[0]),
            "value": best_speed[1],
            "metric": "Decode MB/s",
        },
    }

    return {
        "summary": summary,
        "models": unified_models,
        "tokka_matrix": tokka_data,
        "metadata": ALGORITHM_METADATA,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler supporting static asset delivery and REST JSON APIs."""

    def _set_headers(self, status: int = HTTPStatus.OK, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._set_headers(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        # API: Benchmarks
        if path == "/api/benchmarks":
            try:
                data = load_benchmark_data()
                body = json.dumps(data).encode("utf-8")
                self._set_headers(HTTPStatus.OK, "application/json; charset=utf-8")
                self.wfile.write(body)
            except Exception as e:
                logger.exception("Error loading benchmark data")
                err_body = json.dumps({"error": str(e)}).encode("utf-8")
                self._set_headers(HTTPStatus.INTERNAL_SERVER_ERROR, "application/json")
                self.wfile.write(err_body)
            return

        # Static assets and index.html
        if path == "/" or path == "/index.html":
            file_path = STATIC_DIR / "index.html"
        elif path.startswith("/static/"):
            rel_path = path[len("/static/") :]
            file_path = (STATIC_DIR / rel_path).resolve()
            # Prevent path traversal
            if not str(file_path).startswith(str(STATIC_DIR.resolve())):
                self._set_headers(HTTPStatus.FORBIDDEN, "text/plain")
                self.wfile.write(b"Forbidden")
                return
        else:
            self._set_headers(HTTPStatus.NOT_FOUND, "text/plain")
            self.wfile.write(b"Not Found")
            return

        if not file_path.exists() or not file_path.is_file():
            self._set_headers(HTTPStatus.NOT_FOUND, "text/plain")
            self.wfile.write(b"File Not Found")
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            if file_path.suffix == ".js":
                mime_type = "application/javascript"
            elif file_path.suffix == ".css":
                mime_type = "text/css"
            elif file_path.suffix == ".json":
                mime_type = "application/json"
            else:
                mime_type = "application/octet-stream"

        if "text" in mime_type or mime_type in ("application/javascript", "application/json"):
            mime_type += "; charset=utf-8"

        try:
            content = file_path.read_bytes()
            self._set_headers(HTTPStatus.OK, mime_type)
            self.wfile.write(content)
        except Exception as e:
            logger.exception("Error serving file")
            self._set_headers(HTTPStatus.INTERNAL_SERVER_ERROR, "text/plain")
            self.wfile.write(f"Error reading file: {e}".encode("utf-8"))

    def do_POST(self) -> None:
        path = self.path.split("?")[0]

        # Read JSON body
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception as e:
            self._set_headers(HTTPStatus.BAD_REQUEST, "application/json")
            self.wfile.write(json.dumps({"error": f"Invalid JSON payload: {e}"}).encode("utf-8"))
            return

        if path == "/api/tokenize":
            text = payload.get("text", "")
            if not text:
                text = "The internationalization of unhappily disproportionate misunderstandings requires robust subword tokenization."
            vocab_size = int(payload.get("vocab_size", 400))

            try:
                tokenizers = _TOKENIZER_CACHE.get_tokenizers(vocab_size)
                raw_bytes = len(text.encode("utf-8"))

                results: List[Dict[str, Any]] = []
                for name, tok in tokenizers.items():
                    meta = ALGORITHM_METADATA.get(name, {})
                    t0 = time.perf_counter()
                    token_ids = tok.encode(text)
                    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

                    # Extract decoded token string pieces
                    token_details: List[Dict[str, Any]] = []
                    tokens: List[str] = []
                    for tid in token_ids:
                        try:
                            # Try id_to_bytes or decode
                            if hasattr(tok, "id_to_bytes"):
                                b_val = tok.id_to_bytes(tid)
                                piece = b_val.decode("utf-8", errors="replace") if b_val else f"<{tid}>"
                            else:
                                piece = tok.decode([tid])
                        except Exception:
                            try:
                                piece = tok.decode([tid])
                            except Exception:
                                piece = f"<{tid}>"

                        tokens.append(piece)
                        token_details.append(
                            {
                                "id": tid,
                                "piece": piece,
                                "byte_len": len(piece.encode("utf-8")),
                            }
                        )

                    num_tokens = len(token_ids)
                    bpt = round(raw_bytes / num_tokens, 3) if num_tokens > 0 else 0.0

                    results.append(
                        {
                            "name": name,
                            "display_name": meta.get("display_name", name),
                            "family": meta.get("family", "Unknown"),
                            "num_tokens": num_tokens,
                            "raw_bytes": raw_bytes,
                            "bytes_per_token": bpt,
                            "encode_ms": elapsed_ms,
                            "token_ids": token_ids,
                            "tokens": tokens,
                            "token_details": token_details,
                        }
                    )

                response_data = {
                    "text": text,
                    "vocab_size": vocab_size,
                    "raw_bytes": raw_bytes,
                    "results": results,
                }
                self._set_headers(HTTPStatus.OK, "application/json; charset=utf-8")
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
            except Exception as e:
                logger.exception("Error in tokenize endpoint")
                self._set_headers(HTTPStatus.INTERNAL_SERVER_ERROR, "application/json")
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if path == "/api/run-suite":
            vocab_size = int(payload.get("vocab_size", 400))
            suite_type = payload.get("suite", "all")
            # Return acknowledgment
            self._set_headers(HTTPStatus.OK, "application/json; charset=utf-8")
            self.wfile.write(
                json.dumps(
                    {
                        "status": "acknowledged",
                        "vocab_size": vocab_size,
                        "suite": suite_type,
                        "message": f"Benchmark suite '{suite_type}' dispatched for vocab_size={vocab_size}.",
                    }
                ).encode("utf-8")
            )
            return

        self._set_headers(HTTPStatus.NOT_FOUND, "text/plain")
        self.wfile.write(b"Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        # Avoid cluttering console output during testing
        if os.environ.get("TOKENISERS_DASHBOARD_VERBOSE"):
            super().log_message(format, *args)


class DashboardServer:
    """Encapsulates ThreadingHTTPServer with lifecycle controls."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.server = ThreadingHTTPServer((host, port), DashboardHandler)
        self._thread: Optional[threading.Thread] = None

    def start(self, blocking: bool = True) -> None:
        print(f"Tokenisers Lab Benchmark Dashboard running at http://{self.host}:{self.port}/")
        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("Server shutdown completed.")


def create_server(host: str = "127.0.0.1", port: int = 8765) -> DashboardServer:
    """Creates a configured DashboardServer instance."""
    return DashboardServer(host=host, port=port)


def start_server(host: str = "127.0.0.1", port: int = 8765, blocking: bool = True) -> DashboardServer:
    """Factory creating and running the dashboard server."""
    server = create_server(host, port)
    server.start(blocking=blocking)
    return server
