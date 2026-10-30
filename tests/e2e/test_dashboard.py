"""
End-to-End Tests for Tokenisers Lab Benchmark Intelligence Dashboard & REST APIs.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.request
import urllib.error
from typing import Generator

import pytest

from tokenisers.dashboard import server as server_mod
from tokenisers.dashboard.server import DashboardServer, TokenizerCache, create_server


def find_free_port() -> int:
    """Finds an available ephemeral port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def dashboard_server() -> Generator[str, None, None]:
    """Starts the dashboard server on an ephemeral port and tears it down."""
    port = find_free_port()
    server = create_server(host="127.0.0.1", port=port, offline=True)
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


def _raw(base_url: str, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
    """Send a request with full control over headers; returns (status, headers, body bytes)."""
    port = int(base_url.rsplit(":", 1)[1])
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        hdrs = {"Host": f"127.0.0.1:{port}", **(headers or {})}
        if body is not None:
            hdrs.setdefault("Content-Length", str(len(body)))
        for k, v in hdrs.items():
            if v is not None:
                conn.putheader(k, v)
        conn.endheaders()
        if body is not None:
            conn.send(body)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def _post_json(base_url: str, payload, headers: dict | None = None):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    return _raw(base_url, "POST", "/api/tokenize", body, {"Content-Type": "application/json", **(headers or {})})


def test_api_tokenize_reports_synthetic_corpus(dashboard_server: str) -> None:
    status, _, body = _post_json(dashboard_server, {"text": "hello world", "vocab_size": 320})
    assert status == 200
    corpus = json.loads(body)["corpus"]
    assert corpus["synthetic"] is True
    assert "fixture" in corpus["lexicon_source"]


def test_token_byte_lens_count_token_bytes_not_display_piece(dashboard_server: str) -> None:
    """Partial-UTF-8 tokens display as U+FFFD (3 bytes) but are 1 byte: byte_lens must sum to raw_bytes."""
    text = "héllo 🦀 日本"  # rare multibyte chars: a 320 vocab keeps some as single-byte tokens
    status, _, body = _post_json(dashboard_server, {"text": text, "vocab_size": 320})
    assert status == 200
    data = json.loads(body)
    assert data["raw_bytes"] == len(text.encode("utf-8"))
    partial = 0
    for res in data["results"]:
        lens = [d["byte_len"] for d in res["token_details"]]
        assert sum(lens) == data["raw_bytes"], (res["name"], res["token_details"])
        partial += sum(1 for d in res["token_details"] if d["piece"] == "�" and d["byte_len"] < 3)
    assert partial > 0, "text produced no partial-UTF-8 tokens, so the test checks nothing"


def test_no_wildcard_cors(dashboard_server: str) -> None:
    for method, path in [("GET", "/"), ("GET", "/api/benchmarks")]:
        status, headers, _ = _raw(dashboard_server, method, path)
        assert status == 200
        assert "Access-Control-Allow-Origin" not in headers
    status, headers, _ = _raw(dashboard_server, "OPTIONS", "/api/tokenize", headers={"Origin": "https://evil.example"})
    assert status >= 400 and "Access-Control-Allow-Origin" not in headers


def test_foreign_host_header_rejected(dashboard_server: str) -> None:
    """DNS rebinding: evil.example resolving to 127.0.0.1 must not reach the API."""
    port = dashboard_server.rsplit(":", 1)[1]
    status, _, _ = _raw(dashboard_server, "GET", "/api/benchmarks", headers={"Host": f"evil.example:{port}"})
    assert status == 403
    status, _, _ = _post_json(dashboard_server, {"text": "x"}, headers={"Host": f"evil.example:{port}"})
    assert status == 403
    status, _, _ = _raw(dashboard_server, "GET", "/", headers={"Host": f"localhost:{port}"})
    assert status == 200


def test_cross_origin_post_rejected(dashboard_server: str) -> None:
    port = dashboard_server.rsplit(":", 1)[1]
    status, _, _ = _post_json(dashboard_server, {"text": "x", "vocab_size": 320}, headers={"Origin": "https://evil.example"})
    assert status == 403
    # text/plain is a CORS "simple request": no preflight, so it must be refused outright.
    status, _, _ = _raw(dashboard_server, "POST", "/api/tokenize", b'{"text": "x"}', {"Content-Type": "text/plain"})
    assert status == 415
    status, _, _ = _post_json(dashboard_server, {"text": "x", "vocab_size": 320}, headers={"Origin": f"http://127.0.0.1:{port}"})
    assert status == 200


# The last body nests 30000 arrays (60 KB, under MAX_BODY_BYTES): json.loads raises RecursionError.
@pytest.mark.parametrize(
    "body", [b"abc", b"[1, 2]", b'"text"', b"\xff\xfe", pytest.param(b"[" * 30000 + b"]" * 30000, id="deep-nesting")]
)
def test_malformed_json_is_400(dashboard_server: str, body: bytes) -> None:
    status, _, payload = _post_json(dashboard_server, body)
    assert status == 400
    assert "error" in json.loads(payload)


# Only just-out-of-range values: a huge vocab_size (the finding's 300_000 / 1e9) would train
# 13 tokenizers without bound if validation ever regressed, which can OOM the machine.
@pytest.mark.parametrize(
    "vocab_size",
    [server_mod.MAX_VOCAB_SIZE + 1, server_mod.MIN_VOCAB_SIZE - 1, 0, -5, "400", 400.5, None, True],
)
def test_vocab_size_outside_allow_list_is_400(dashboard_server: str, vocab_size) -> None:
    status, _, payload = _post_json(dashboard_server, {"text": "abc", "vocab_size": vocab_size})
    assert status == 400
    assert "vocab_size" in json.loads(payload)["error"]


def test_body_and_text_size_caps(dashboard_server: str) -> None:
    status, _, _ = _post_json(dashboard_server, {"text": "a" * (server_mod.MAX_BODY_BYTES + 1)})
    assert status == 413
    # Raw UTF-8 keeps this body under MAX_BODY_BYTES, so only the text-length cap can fire.
    body = json.dumps({"text": "é" * (server_mod.MAX_TEXT_CHARS + 1)}, ensure_ascii=False).encode("utf-8")
    assert len(body) <= server_mod.MAX_BODY_BYTES
    status, _, payload = _post_json(dashboard_server, body)
    assert status == 413 and "text exceeds" in json.loads(payload)["error"]
    status, _, _ = _raw(dashboard_server, "POST", "/api/tokenize", None, {"Content-Type": "application/json"})
    assert status == 411


def test_static_path_traversal_uses_real_containment(dashboard_server: str, tmp_path, monkeypatch) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "ok.js").write_text("ok", encoding="utf-8")
    sibling = tmp_path / "static_secrets"  # shares the string prefix ".../static"
    sibling.mkdir()
    (sibling / "key.txt").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(server_mod, "STATIC_DIR", static)

    status, _, body = _raw(dashboard_server, "GET", "/static/ok.js")
    assert status == 200 and body == b"ok"
    status, _, body = _raw(dashboard_server, "GET", "/static/../static_secrets/key.txt")
    assert status == 403 and b"secret" not in body
    status, _, _ = _raw(dashboard_server, "GET", "/static/../../etc/passwd")
    assert status == 403


def test_run_suite_stub_removed(dashboard_server: str) -> None:
    status, _, _ = _post_json(dashboard_server, {"suite": "intrinsic", "vocab_size": 320})
    assert status == 200  # /api/tokenize itself still works
    status, _, _ = _raw(dashboard_server, "POST", "/api/run-suite", b"{}", {"Content-Type": "application/json"})
    assert status == 404


def test_server_refuses_non_loopback_bind() -> None:
    with pytest.raises(ValueError):
        DashboardServer(host="0.0.0.0", port=find_free_port())


def test_tokenizer_cache_is_bounded_lru(monkeypatch) -> None:
    cache = TokenizerCache(offline=True, max_entries=2)
    trained = []
    monkeypatch.setattr(cache, "_train", lambda v: trained.append(v) or {"v": v})
    for v in (300, 310, 300, 320, 310):
        assert cache.get_tokenizers(v) == {"v": v}
    assert list(cache._cache) == [320, 310]
    assert trained == [300, 310, 320, 310]  # 310 was evicted by 320, 300 reused once


# (fragment the formula must show, fragment it must not) per scorer, from the Rust scorers.
FORMULA_FACTS = {
    "wordpiece_pmi": ([r"(\mathrm{count}(u) + \alpha)", r"\alpha = 1"], []),          # Laplace-smoothed
    "wp_bpe_base": ([r"{\mathrm{freq}(u) \cdot \mathrm{freq}(v)}"], [r"\alpha", r"\theta"]),
    "wp_bpe_ft": ([r"{\mathrm{freq}(u) \cdot \mathrm{freq}(v)}", r"\theta = 2"], [r"\alpha"]),  # ratio + cutoff
    "wp_bpe_scp": ([r"(\mathrm{freq}(uv) - d)_+", r"\sqrt{\mathrm{freq}(u)\,\mathrm{freq}(v)}"], ["P(uv)^2"]),
    "wp_bpe_npmi": ([r"\mathrm{freq}(uv)^{\alpha}", r"-\ln P(uv)"], [r"\in [-1, +1]"]),
    "wp_bpe_ppig": ([r"\max(0, \mathrm{PMI}(u, v))", r"(|u| + |v|)^{\eta}"], [r"\mathcal{H}"]),
    "mdl_bits": ([r"8\,|uv|", r"N \log_2 N"], [r"+ \Delta L"]),
    "length_penalized": ([r"{(|u| + |v|)^\beta}"], []),
    "morpheme_bpe": ([r"e^{\lambda M(u, v)}", r"n_{\mathrm{intra}} - n_{\mathrm{cross}}"], [r"1 + \lambda"]),
}


@pytest.mark.parametrize("name", sorted(FORMULA_FACTS))
def test_formula_matches_implemented_scorer(name: str) -> None:
    formula = server_mod.ALGORITHM_METADATA[name]["formula"]
    must, must_not = FORMULA_FACTS[name]
    for fragment in must:
        assert fragment in formula, (name, fragment)
    for fragment in must_not:
        assert fragment not in formula, (name, fragment)


def test_static_page_formulas_are_the_api_formulas() -> None:
    """index.html's hard-coded formulas must be copies of ALGORITHM_METADATA (one source of truth)."""
    import re

    html = (server_mod.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    shown = re.findall(r"\$\$(.+?)\$\$", html)
    assert len(shown) >= 6
    known = {m["formula"] for m in server_mod.ALGORITHM_METADATA.values()}
    assert [f for f in shown if f not in known] == []
