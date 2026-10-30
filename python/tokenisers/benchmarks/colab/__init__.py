"""
Extrinsic proxy-LM benchmark: a small causal Transformer is trained per tokenizer under a fixed
UTF-8 byte budget and scored by validation bits per byte (BPB).

The model, trainer, evaluator and job runner live in ``colab/lm.py``, which is imported locally
and uploaded unchanged to Colab, so local and remote runs share one implementation.
``ColabOrchestrator`` runs a job directory on a Colab VM through the ``colab`` CLI (>= 0.7.5).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
import uuid

from tokenisers.benchmarks.colab.lm import (
    BpbEvaluator,
    ByteBudgetTrainer,
    ProxyLanguageModel,
    run_job,
)

LM_SOURCE = Path(__file__).with_name("lm.py")
REMOTE_DIR = "/content"
# 0.6.0's `exec` fails with AttributeError (jupyter_kernel_client has no KernelClient).
MIN_CLI_VERSION: Tuple[int, int, int] = (0, 7, 5)
UPGRADE_HINT = "uv tool upgrade google-colab-cli"
STATUS_MARKER = "TOKENISERS_JOB "
# One poll: a short `colab exec`. A call that hangs (e.g. "Connection was lost") is killed after
# EXEC_TIMEOUT seconds and counts as a failed poll.
EXEC_TIMEOUT = 300.0

# Run in the Colab kernel on every poll (``colab exec`` exits 0 even when the code raises, so the
# state is read from the STATUS_MARKER line, never from the exit code). Idempotent: the first call
# launches ``python lm.py`` detached (own session, output in job.log, pid in job.pid) and returns at
# once; later calls report "running", or the final status.json that lm.main writes ("ok"/"error"),
# or "died" when the process is gone without one. A poll lost to a dropped connection is retried.
CONTROL_SCRIPT = """\
import json, os, subprocess, sys
d = {remote_dir!r}
pid_path, status_path, log_path = (os.path.join(d, f) for f in ("job.pid", "status.json", "job.log"))

def alive(pid):
    try:
        with open(f"/proc/{{pid}}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False

if not os.path.exists(pid_path):
    with open(log_path, "ab") as log:
        p = subprocess.Popen([sys.executable, "-u", os.path.join(d, "lm.py"), d], cwd=d, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with open(pid_path, "w") as f:
        f.write(str(p.pid))
    out = {{"state": "launched", "pid": p.pid}}
else:
    with open(pid_path) as f:
        pid = int(f.read())
    running = alive(pid)  # before reading the status: lm.py writes status.json, then exits
    if os.path.exists(status_path):
        with open(status_path) as f:
            out = json.load(f)
    else:
        out = {{"state": "running" if running else "died", "pid": pid}}
try:
    with open(log_path, "rb") as f:
        f.seek(max(0, os.path.getsize(log_path) - 4000))
        out["log_tail"] = f.read().decode("utf-8", "replace")
except OSError:
    out["log_tail"] = ""
print({marker!r} + json.dumps(out))
"""


class ColabError(RuntimeError):
    """A ``colab`` CLI step or the remote job failed (the message carries the remote traceback)."""


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
        """Runs one CLI command; returns its stdout + stderr. A non-zero exit raises ColabError, but
        exit 0 does not mean the remote code succeeded (see CONTROL_SCRIPT)."""
        cmd = [self.colab_bin, *args]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ColabError(f"`colab {args[0]}` failed: {e}") from e
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip()[-2000:]
            raise ColabError(f"`colab {args[0]}` exited with {proc.returncode}: {tail}")
        return proc.stdout + proc.stderr

    def check_cli_version(self) -> Tuple[int, ...]:
        """Raises ColabError unless ``colab version`` reports >= MIN_CLI_VERSION."""
        need = ".".join(map(str, MIN_CLI_VERSION))
        try:
            out = self._colab("version", timeout=60)
        except ColabError as e:
            raise ColabError(f"{e}\nthe benchmark needs colab CLI >= {need}; upgrade with `{UPGRADE_HINT}`") from e
        m = re.search(r"Version:\s*(\d+)\.(\d+)\.(\d+)", out)
        if not m:
            raise ColabError(f"cannot read the colab CLI version from `colab version` ({out.strip()[-200:]!r}); "
                             f"the benchmark needs >= {need}; upgrade with `{UPGRADE_HINT}`")
        version = tuple(int(g) for g in m.groups())
        if version < MIN_CLI_VERSION:
            raise ColabError(f"colab CLI {'.'.join(m.groups())} is too old: the benchmark needs >= {need} "
                             f"(0.6.0's `exec` is broken); upgrade with `{UPGRADE_HINT}`")
        return version

    @contextlib.contextmanager
    def managed_session(self, session_name: str, gpu: Optional[str] = "T4") -> Iterator[ColabSession]:
        """Creates a session and always stops it afterwards, also when ``colab new`` fails or times
        out (the VM may exist even though the client gave up)."""
        session = ColabSession(session_id=session_name, active=True)
        try:
            if not self.mock:
                self._colab("new", "-s", session_name, *(["--gpu", gpu] if gpu else []))
            yield session
        finally:
            session.active = False
            if not self.mock:
                try:
                    self._colab("stop", "-s", session_name, timeout=120)
                except ColabError:
                    pass

    def _poll(self, name: str, control: Path) -> Dict[str, Any]:
        out = self._colab("exec", "-s", name, "-f", str(control), "--timeout", "120", timeout=EXEC_TIMEOUT)
        for line in out.splitlines():
            if line.startswith(STATUS_MARKER):
                return json.loads(line[len(STATUS_MARKER):])
        raise ColabError(f"`colab exec` of the job control script printed no status:\n{out.strip()[-3000:]}")

    def _wait(self, name: str, control: Path, timeout: float, poll_interval: float,
              max_poll_failures: int) -> Dict[str, Any]:
        """Launches the job (first poll) and polls until it is ok/error/died; raises ColabError
        after ``max_poll_failures`` consecutive failed polls or when ``timeout`` seconds pass."""
        deadline = time.monotonic() + timeout
        failures = 0
        last: Dict[str, Any] = {}
        while True:
            try:
                last = self._poll(name, control)
                failures = 0
            except ColabError as e:
                failures += 1
                if failures >= max_poll_failures:
                    raise ColabError(f"{failures} consecutive job polls failed; last error: {e}") from e
            if last.get("state") in ("ok", "error", "died"):
                return last
            if time.monotonic() >= deadline:
                raise ColabError(f"the remote job did not finish within {timeout:.0f} s; "
                                 f"log tail:\n{last.get('log_tail', '')}")
            time.sleep(poll_interval)

    def run_job(
        self,
        job_dir: Path,
        gpu: Optional[str] = "T4",
        timeout: float = 6 * 3600,
        poll_interval: float = 60.0,
        max_poll_failures: int = 5,
    ) -> Dict[str, Any]:
        """
        Checks the CLI version, uploads ``lm.py`` and the job files, starts ``lm.py`` detached on
        the VM and polls it every ``poll_interval`` s with short ``colab exec`` calls (no exec stays
        open for the whole job), then downloads result.json and returns it with
        ``backend.kind = "colab"``. A remote exception, a job that dies or runs past ``timeout``,
        or any failed CLI step raises ColabError (with the remote traceback / log tail); the
        session is always stopped. There is no silent local fallback.
        """
        job_dir = Path(job_dir)
        if not self.mock:
            self.check_cli_version()
        name = f"tokenisers-bpb-{uuid.uuid4().hex[:8]}"
        files: List[Path] = [LM_SOURCE, job_dir / "config.json", *sorted(job_dir.glob("*.npz"))]
        control = job_dir / "control.py"
        control.write_text(CONTROL_SCRIPT.format(remote_dir=REMOTE_DIR, marker=STATUS_MARKER), encoding="utf-8")
        local_result = job_dir / "result.json"
        local_result.unlink(missing_ok=True)
        with self.managed_session(name, gpu):
            for f in files:
                self._colab("upload", "-s", name, str(f), f"{REMOTE_DIR}/{f.name}", timeout=3600)
            status = self._wait(name, control, timeout, poll_interval, max_poll_failures)
            if status["state"] != "ok":
                detail = status.get("traceback") or f"no status.json; log tail:\n{status.get('log_tail', '')}"
                raise ColabError(f"the remote job failed ({status['state']}):\n{detail}")
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
