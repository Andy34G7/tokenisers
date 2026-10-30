"""
Extrinsic proxy-LM benchmark: a small causal Transformer is trained per tokenizer under a fixed
UTF-8 byte budget and scored by validation bits per byte (BPB).

The model, trainer, evaluator and job runner live in ``colab/lm.py``, which is imported locally
and uploaded unchanged to Colab, so local and remote runs share one implementation.
``ColabOrchestrator`` runs a job directory on a Colab VM through the ``colab`` CLI.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterator, List, Optional
import uuid

from tokenisers.benchmarks.colab.lm import (
    BpbEvaluator,
    ByteBudgetTrainer,
    ProxyLanguageModel,
    run_job,
)

LM_SOURCE = Path(__file__).with_name("lm.py")
REMOTE_DIR = "/content"


class ColabError(RuntimeError):
    """A ``colab`` CLI step failed; nothing was computed remotely."""


@dataclass
class ColabSession:
    session_id: str
    active: bool = True


class ColabOrchestrator:
    """Runs proxy-LM jobs on Google Colab via the ``colab`` CLI. ``mock=True`` never calls the CLI."""

    def __init__(self, mock: bool = False, colab_bin: Optional[str] = None) -> None:
        self.mock = mock
        self.colab_bin = colab_bin or str(Path.home() / ".local" / "bin" / "colab")

    def _colab(self, *args: str, timeout: Optional[float] = 600) -> str:
        cmd = [self.colab_bin, *args]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ColabError(f"`colab {args[0]}` failed: {e}") from e
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip()[-2000:]
            raise ColabError(f"`colab {args[0]}` exited with {proc.returncode}: {tail}")
        return proc.stdout

    @contextlib.contextmanager
    def managed_session(self, session_name: str, gpu: Optional[str] = "T4") -> Iterator[ColabSession]:
        """Creates a session (raising ColabError if that fails) and always stops it afterwards."""
        session = ColabSession(session_id=session_name, active=True)
        if not self.mock:
            self._colab("new", "-s", session_name, *(["--gpu", gpu] if gpu else []))
        try:
            yield session
        finally:
            session.active = False
            if not self.mock:
                try:
                    self._colab("stop", "-s", session_name, timeout=120)
                except ColabError:
                    pass

    def run_job(self, job_dir: Path, gpu: Optional[str] = "T4", timeout: float = 6 * 3600) -> Dict[str, Any]:
        """
        Uploads ``lm.py`` and the job files, runs ``lm.run_job`` in the Colab kernel, downloads
        result.json and returns it with ``backend.kind = "colab"``. Any failed step raises
        ColabError; there is no silent local fallback.
        """
        job_dir = Path(job_dir)
        name = f"tokenisers-bpb-{uuid.uuid4().hex[:8]}"
        files: List[Path] = [LM_SOURCE, job_dir / "config.json", *sorted(job_dir.glob("*.npz"))]
        driver = job_dir / "driver.py"
        driver.write_text(
            "import sys\n"
            f"sys.path.insert(0, {REMOTE_DIR!r})\n"
            "import lm\n"
            f"lm.run_job({REMOTE_DIR!r})\n",
            encoding="utf-8",
        )
        local_result = job_dir / "result.json"
        local_result.unlink(missing_ok=True)
        with self.managed_session(name, gpu):
            for f in files:
                self._colab("upload", "-s", name, str(f), f"{REMOTE_DIR}/{f.name}", timeout=3600)
            self._colab("exec", "-s", name, "-f", str(driver), "--timeout", str(timeout), timeout=timeout + 300)
            self._colab("download", "-s", name, f"{REMOTE_DIR}/result.json", str(local_result), timeout=600)
        if not local_result.is_file():
            raise ColabError("the Colab job produced no result.json")
        result = json.loads(local_result.read_text(encoding="utf-8"))
        result["backend"].update({"kind": "colab", "requested_gpu": gpu, "session": name})
        return result


def run_local(job_dir: Path, device: Optional[str] = None) -> Dict[str, Any]:
    result = run_job(str(job_dir), device=device)
    result["backend"]["kind"] = "local"
    return result


__all__ = [
    "ProxyLanguageModel",
    "ByteBudgetTrainer",
    "BpbEvaluator",
    "ColabError",
    "ColabOrchestrator",
    "ColabSession",
    "run_job",
    "run_local",
]
