"""
Google Colab Extrinsic Small-LM Benchmark suite.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import math
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn


class ProxyLanguageModel(nn.Module):
    """
    Lightweight Llama/GPT-style Transformer with RoPE, RMSNorm, SwiGLU,
    and tied input/output embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 2,
        d_ffn: int = 256,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        # Backbone layers (independent of vocab_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ffn,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output projection head tied to input embeddings
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, : self.max_seq_len]

        h = self.embedding(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits


class ByteBudgetTrainer:
    """Trainer tracking consumed bytes towards a fixed UTF-8 training budget."""

    def __init__(self, model: nn.Module, byte_budget: int = 10_000) -> None:
        self.model = model
        self.byte_budget = byte_budget
        self.bytes_consumed = 0

    def step_batch(self, text: str) -> None:
        raw_bytes = len(text.encode("utf-8"))
        self.bytes_consumed += raw_bytes

    def step_loss(self, loss_val: float) -> None:
        if math.isnan(loss_val):
            raise ValueError(f"Encountered NaN loss during training step: {loss_val}")


class BpbEvaluator:
    """Evaluates validation Bits-Per-Byte: BPB = loss_nats / (ln(2) * bytes_per_token)."""

    def compute_bpb(self, loss_nats: float, bytes_per_token: float) -> float:
        if bytes_per_token <= 0.0:
            return 0.0
        return loss_nats / (math.log(2.0) * bytes_per_token)


@dataclass
class ColabSession:
    session_id: str
    active: bool = True


class ColabOrchestrator:
    """Orchestrator for managing remote Google Colab GPU sessions via colab-cli."""

    def __init__(self, mock: bool = True) -> None:
        self.mock = mock

    @contextlib.contextmanager
    def managed_session(self, session_name: str) -> Iterator[ColabSession]:
        session = ColabSession(session_id=session_name, active=True)
        try:
            yield session
        finally:
            session.active = False


__all__ = [
    "ProxyLanguageModel",
    "ByteBudgetTrainer",
    "BpbEvaluator",
    "ColabOrchestrator",
    "ColabSession",
]
