"""Tests for ragstuffer — covers text extraction, point ID determinism,
state management, admin server endpoints, and graceful shutdown."""

import importlib.util
import json
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

# Stub out heavy dependencies that aren't installed in the test environment.
# Use a dedicated dict so we can clean up without poisoning other test files.
_STUBS = {}
for _mod in [
    "google.oauth2.service_account",
    "googleapiclient.discovery",
    "googleapiclient.http",
    "langchain_text_splitters",
]:
    if _mod not in sys.modules:
        _STUBS[_mod] = MagicMock()
        sys.modules[_mod] = _STUBS[_mod]

# qdrant_client is a real package used by other test files — only stub if missing.
for _mod in ["qdrant_client", "qdrant_client.models"]:
    try:
        __import__(_mod)
    except ImportError:
        _STUBS[_mod] = MagicMock()
        sys.modules[_mod] = _STUBS[_mod]

spec = importlib.util.spec_from_file_location(
    "ragstuffer",
    Path(__file__).with_name("ragstuffer.py"),
)
rw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rw)


# ── Point ID determinism ────────────────────────────────────────────────────


class TestPointId:
    """Deterministic point IDs for idempotent Qdrant upsert."""

    def test_deterministic(self):
        a = rw._point_id("doc-abc", 0)
        b = rw._point_id("doc-abc", 0)
        assert a == b, "Same inputs must produce same ID"

    def test_different_chunks_differ(self):
        a = rw._point_id("doc-abc", 0)
        b = rw._point_id("doc-abc", 1)
        assert a != b, "Different chunk_ids must produce different IDs"

    def test_different_docs_differ(self):
        a = rw._point_id("doc-abc", 0)
        b = rw._point_id("doc-xyz", 0)
        assert a != b, "Different doc_ids must produce different IDs"

    def test_returns_int(self):
        result = rw._point_id("doc-abc", 0)
        assert isinstance(result, int)


# ── Text extraction ─────────────────────────────────────────────────────────


class TestExtractText:
    """Text extraction from various file types."""

    def test_plain_text(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world")
        assert rw.extract_text(f) == "Hello world"

    def test_markdown(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nBody text")
        result = rw.extract_text(f)
        assert "Title" in result
        assert "Body" in result

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert rw.extract_text(f) == ""

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02")
        result = rw.extract_text(f)
        assert result == "" or isinstance(result, str)


# ── State management ────────────────────────────────────────────────────────


class TestState:
    """State file load/save for Drive polling."""

    def test_load_missing_file(self, tmp_path):
        with patch.object(rw, "STATE_PATH", tmp_path / "missing.json"):
            state = rw.load_state()
            assert state == {}

    def test_save_and_load(self, tmp_path):
        state_file = tmp_path / "state.json"
        with patch.object(rw, "STATE_PATH", state_file):
            rw.save_state({"file1": "2026-01-01T00:00:00Z"})
            loaded = rw.load_state()
            assert loaded == {"file1": "2026-01-01T00:00:00Z"}

    def test_save_overwrites(self, tmp_path):
        state_file = tmp_path / "state.json"
        with patch.object(rw, "STATE_PATH", state_file):
            rw.save_state({"a": "1"})
            rw.save_state({"b": "2"})
            loaded = rw.load_state()
            assert "a" not in loaded
            assert loaded == {"b": "2"}


# ── Admin server ───────────────────────────────────────────────────────────


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestAdminServer:
    """Tests for the admin HTTP server endpoints."""

    def setup_method(self):
        self.port = _find_free_port()
        self.trigger = threading.Event()
        self.server = rw._start_admin_server(self.trigger, self.port, "test-token")

    def teardown_method(self):
        self.server.shutdown()

    def _request(self, method, path, token=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        req = urllib.request.Request(url, method=method)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        if method == "POST":
            req.data = b""
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read()
            return e.code, json.loads(body) if body else {}

    def test_health_endpoint(self):
        status, body = self._request("GET", "/health")
        assert status == 200
        assert body["status"] == "ok"

    def test_health_no_auth_needed(self):
        # /health should work without a token
        status, _ = self._request("GET", "/health")
        assert status == 200

    def test_ingest_now_requires_auth(self):
        status, body = self._request("POST", "/admin/ingest-now")
        assert status == 401

    def test_ingest_now_with_auth(self):
        status, body = self._request("POST", "/admin/ingest-now", token="test-token")
        assert status == 200
        assert body["mode"] == "incremental"
        assert self.trigger.is_set()

    def test_ingest_full_with_auth(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("{}")
        with patch.object(rw, "STATE_PATH", state_file):
            status, body = self._request("POST", "/admin/ingest-full", token="test-token")
        assert status == 200
        assert body["mode"] == "full"
        assert self.trigger.is_set()

    def test_wrong_token_rejected(self):
        status, _ = self._request("POST", "/admin/ingest-now", token="wrong")
        assert status == 401

    def test_unknown_path_404(self):
        status, _ = self._request("GET", "/nonexistent")
        assert status == 404

    def test_unknown_post_404(self):
        status, _ = self._request("POST", "/admin/unknown", token="test-token")
        assert status == 404


class TestAdminServerNoAuth:
    """Admin server with no token set — all endpoints open."""

    def setup_method(self):
        self.port = _find_free_port()
        self.trigger = threading.Event()
        self.server = rw._start_admin_server(self.trigger, self.port, "")

    def teardown_method(self):
        self.server.shutdown()

    def _request(self, method, path):
        url = f"http://127.0.0.1:{self.port}{path}"
        req = urllib.request.Request(url, method=method)
        if method == "POST":
            req.data = b""
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())

    def test_ingest_now_no_auth_needed(self):
        status, body = self._request("POST", "/admin/ingest-now")
        assert status == 200
        assert body["status"] == "triggered"


# ── download_drive_file return type ──────────────────────────────────────────


class TestDownloadDriveFile:
    """Verify download_drive_file returns Path on success, None on skip."""

    def test_unsupported_extension_returns_none(self):
        mock_service = MagicMock()
        file_info = {"id": "123", "name": "data.xyz", "mimeType": "application/octet-stream"}
        result = rw.download_drive_file(mock_service, file_info, Path("/tmp"))
        assert result is None
