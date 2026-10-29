"""
End-to-End Tests for Tokenisers Lab Benchmark Intelligence Dashboard & REST APIs.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
import urllib.error
from typing import Generator

import pytest

from tokenisers.dashboard.server import DashboardServer, create_server


def find_free_port() -> int:
    """Finds an available ephemeral port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def dashboard_server() -> Generator[str, None, None]:
    """Starts the dashboard server on an ephemeral port and tears it down."""
    port = find_free_port()
    server = create_server(host="127.0.0.1", port=port)
    server.start(blocking=False)

    # Wait for server to bind and listen
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 5.0
    connected = False
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/", timeout=1.0) as resp:
                if resp.status == 200:
                    connected = True
                    break
        except Exception:
            time.sleep(0.05)

    assert connected, "Failed to start DashboardServer within deadline"

    yield base_url

    server.stop()


def test_dashboard_index_html(dashboard_server: str) -> None:
    """Tests that GET / serves the interactive dashboard HTML."""
    req = urllib.request.Request(f"{dashboard_server}/")
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        assert resp.status == 200
        content_type = resp.headers.get("Content-Type", "")
        assert "text/html" in content_type
        body = resp.read().decode("utf-8")
        assert "Tokenisers Lab" in body
        assert "/static/app.js" in body
        assert 'id="quadrant-chart"' in body


def test_dashboard_static_app_js(dashboard_server: str) -> None:
    """Tests that GET /static/app.js serves valid JavaScript."""
    req = urllib.request.Request(f"{dashboard_server}/static/app.js")
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        assert resp.status == 200
        content_type = resp.headers.get("Content-Type", "")
        assert "application/javascript" in content_type or "text/javascript" in content_type
        body = resp.read().decode("utf-8")
        assert "renderQuadrantChart" in body
        assert "runLiveTokenize" in body


def test_api_benchmarks(dashboard_server: str) -> None:
    """Tests that GET /api/benchmarks returns consolidated benchmark JSON."""
    req = urllib.request.Request(f"{dashboard_server}/api/benchmarks")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))

        # Verify summary KPIs
        assert "summary" in data
        assert "best_downstream_quality" in data["summary"]
        assert "highest_information_efficiency" in data["summary"]
        assert "best_compression" in data["summary"]

        # Verify models dictionary
        assert "models" in data
        models = data["models"]
        for key in ["bpe_frequency", "wp_bpe_ft", "wp_bpe_pig", "unigram_lm"]:
            assert key in models, f"Model {key} missing from benchmarks"
            m = models[key]
            assert "display_name" in m
            assert "family" in m
            assert "formula" in m

        # Verify tokka matrix
        assert "tokka_matrix" in data
        # Verify metadata
        assert "metadata" in data
        assert "wp_bpe_npmi" in data["metadata"]
        assert "wp_bpe_scp" in data["metadata"]
        assert "wp_bpe_ppig" in data["metadata"]


def test_api_tokenize_live(dashboard_server: str) -> None:
    """Tests that POST /api/tokenize executes multi-tokenizer encoding."""
    payload = {
        "text": "The internationalization of unhappily disproportionate misunderstandings.",
        "vocab_size": 320,
    }
    req = urllib.request.Request(
        f"{dashboard_server}/api/tokenize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))

        assert data["vocab_size"] == 320
        assert data["raw_bytes"] > 0
        assert "results" in data
        results = data["results"]
        assert len(results) >= 8, f"Expected at least 8 tokenizers, got {len(results)}"

        for res in results:
            assert "name" in res
            assert "num_tokens" in res
            assert res["num_tokens"] > 0
            assert "bytes_per_token" in res
            assert "token_ids" in res
            assert "tokens" in res
            assert "token_details" in res
            assert len(res["token_ids"]) == res["num_tokens"]
            assert len(res["tokens"]) == res["num_tokens"]


def test_api_run_suite(dashboard_server: str) -> None:
    """Tests that POST /api/run-suite accepts benchmark suite runs."""
    payload = {"suite": "intrinsic", "vocab_size": 320}
    req = urllib.request.Request(
        f"{dashboard_server}/api/run-suite",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=3.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "acknowledged"
        assert data["vocab_size"] == 320
