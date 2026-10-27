"""
Google Colab Extrinsic Small-LM Benchmark suite.
Trains a controlled Llama/GPT-style Transformer proxy language model under a fixed
UTF-8 byte budget and evaluates validation Bits-Per-Byte (BPB) across English,
Multilingual, and Code splits.
"""

from __future__ import annotations

import contextlib
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyLanguageModel(nn.Module):
    """
    Lightweight Llama/GPT-style causal Transformer with tied input/output embeddings.
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
            seq_len = self.max_seq_len

        h = self.embedding(input_ids)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=input_ids.device
        )
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask, is_causal=True)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits


class ByteBudgetTrainer:
    """Trainer tracking consumed bytes towards a fixed UTF-8 training budget."""

    def __init__(
        self,
        model: nn.Module,
        byte_budget: int = 10_000,
        lr: float = 3e-3,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.byte_budget = byte_budget
        self.bytes_consumed = 0
        self.tokens_consumed = 0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

    def step_batch(self, text: str) -> None:
        raw_bytes = len(text.encode("utf-8"))
        self.bytes_consumed += raw_bytes

    def step_loss(self, loss_val: float) -> None:
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise ValueError(f"Encountered NaN/Inf loss during training step: {loss_val}")

    def train_on_token_stream(
        self,
        token_ids: List[int],
        total_utf8_bytes: int,
        seq_len: int = 64,
        batch_size: int = 8,
    ) -> List[Dict[str, float]]:
        """
        Trains the proxy language model on a stream of token IDs up to `self.byte_budget`
        UTF-8 bytes, returning the training curve (bytes_consumed, tokens_consumed, loss_nats).
        """
        if len(token_ids) < seq_len + 1:
            repeats = ((seq_len + 2) // max(1, len(token_ids))) + 2
            token_ids = (token_ids * repeats)[: seq_len + 2]

        bytes_per_token = max(1e-3, total_utf8_bytes / max(1, len(token_ids)))
        self.model.train()
        history: List[Dict[str, float]] = []

        n_tokens = len(token_ids)
        cursor = 0
        step = 0

        while self.bytes_consumed < self.byte_budget:
            batch_seqs = []
            for _ in range(batch_size):
                if cursor + seq_len + 1 > n_tokens:
                    cursor = 0
                batch_seqs.append(token_ids[cursor : cursor + seq_len + 1])
                cursor += seq_len

            tensor = torch.tensor(batch_seqs, dtype=torch.long, device=self.device)
            x = tensor[:, :-1]
            y = tensor[:, 1:]

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss_val = float(loss.item())
            self.step_loss(loss_val)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            batch_tokens = batch_size * seq_len
            batch_bytes = int(round(batch_tokens * bytes_per_token))
            self.tokens_consumed += batch_tokens
            self.bytes_consumed += max(1, batch_bytes)
            step += 1

            history.append({
                "step": step,
                "bytes_consumed": self.bytes_consumed,
                "tokens_consumed": self.tokens_consumed,
                "train_loss_nats": loss_val,
                "train_bpb": loss_val / (math.log(2.0) * bytes_per_token),
            })

        return history


class BpbEvaluator:
    """Evaluates validation Bits-Per-Byte: BPB = loss_nats / (ln(2) * bytes_per_token)."""

    def compute_bpb(self, loss_nats: float, bytes_per_token: float) -> float:
        if bytes_per_token <= 0.0:
            return 0.0
        return loss_nats / (math.log(2.0) * bytes_per_token)

    @torch.no_grad()
    def evaluate_validation_bpb(
        self,
        model: nn.Module,
        token_ids: List[int],
        utf8_bytes: int,
        seq_len: int = 64,
    ) -> Dict[str, float]:
        """Computes exact validation NLL (nats/token) and Bits-Per-Byte (BPB) on a validation stream."""
        if not token_ids or utf8_bytes <= 0:
            return {"loss_nats": 0.0, "bytes_per_token": 0.0, "bpb": 0.0}

        device = next(model.parameters()).device
        model.eval()

        if len(token_ids) < 2:
            token_ids = token_ids * 2

        total_nll = 0.0
        total_pred_tokens = 0
        for i in range(0, len(token_ids) - 1, seq_len):
            chunk = token_ids[i : i + seq_len + 1]
            if len(chunk) < 2:
                continue
            t = torch.tensor([chunk], dtype=torch.long, device=device)
            logits = model(t[:, :-1])
            targets = t[:, 1:]
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_pred_tokens += targets.numel()

        avg_loss_nats = total_nll / max(1, total_pred_tokens)
        bpt = utf8_bytes / max(1, len(token_ids))
        bpb = self.compute_bpb(avg_loss_nats, bpt)
        return {
            "loss_nats": avg_loss_nats,
            "bytes_per_token": bpt,
            "bpb": bpb,
        }


@dataclass
class ColabSession:
    session_id: str
    active: bool = True


class ColabOrchestrator:
    """Orchestrator for managing remote Google Colab GPU sessions via colab-cli."""

    def __init__(self, mock: bool = True, colab_bin: Optional[str] = None) -> None:
        self.mock = mock
        self.colab_bin = colab_bin or str(Path.home() / ".local" / "bin" / "colab")

    @contextlib.contextmanager
    def managed_session(self, session_name: str, gpu: str = "T4") -> Iterator[ColabSession]:
        session = ColabSession(session_id=session_name, active=True)
        if not self.mock:
            subprocess.run(
                [self.colab_bin, "--auth=oauth2", "new", "-s", session_name, "--gpu", gpu],
                check=False,
                capture_output=True,
                text=True,
            )
        try:
            yield session
        finally:
            session.active = False
            if not self.mock:
                subprocess.run(
                    [self.colab_bin, "--auth=oauth2", "stop", "-s", session_name],
                    check=False,
                    capture_output=True,
                    text=True,
                )

    def run_ephemeral_script(
        self,
        script_code: str,
        gpu: Optional[str] = "T4",
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """
        Executes a self-contained Python training script on Google Colab via `colab run`
        (which provisions the VM, executes the script, and automatically stops the VM).
        Falls back cleanly to CPU Colab VM or local PyTorch if GPU entitlement is unavailable.
        """
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(script_code)
            script_path = f.name

        try:
            if not self.mock and Path(self.colab_bin).exists():
                cmd = [self.colab_bin, "--auth=oauth2", "run"]
                if gpu:
                    cmd.extend(["--gpu", gpu])
                cmd.append(script_path)
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if proc.returncode != 0 and gpu:
                    # Fallback to Colab CPU runtime if GPU tier is unavailable
                    cmd_cpu = [self.colab_bin, "--auth=oauth2", "run", script_path]
                    proc = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=timeout)
                if proc.returncode == 0 and proc.stdout.strip():
                    for line in reversed(proc.stdout.strip().splitlines()):
                        line = line.strip()
                        if line.startswith("{") and line.endswith("}"):
                            return json.loads(line)

            # Local fallback execution
            proc_local = subprocess.run(
                ["uv", "run", "python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            for line in reversed(proc_local.stdout.strip().splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    return json.loads(line)
            raise RuntimeError(f"Failed to parse benchmark output: {proc_local.stderr}")
        finally:
            Path(script_path).unlink(missing_ok=True)


__all__ = [
    "ProxyLanguageModel",
    "ByteBudgetTrainer",
    "BpbEvaluator",
    "ColabOrchestrator",
    "ColabSession",
]
