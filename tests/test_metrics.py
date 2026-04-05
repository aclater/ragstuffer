"""Tests for ragstuffer metrics endpoint."""

import importlib.util
import json
import socket
import sys
import threading
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock

_STUBS = {}
for _mod in [
    "google.oauth2.service_account",
    "googleapiclient.discovery",
    "googleapiclient.http",
    "langchain_text_splitters",
    "qdrant_client",
    "qdrant_client.models",
]:
    if _mod not in sys.modules:
        _STUBS[_mod] = MagicMock()
        sys.modules[_mod] = _STUBS[_mod]

spec = importlib.util.spec_from_file_location(
    "ragstuffer",
    Path(__file__).parent.parent / "ragstuffer" / "__init__.py",
)
ragstuffer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ragstuffer)


def _pick_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        trigger = threading.Event()
        port = _pick_port()
        server = ragstuffer._start_admin_server(trigger, port, "")
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
            assert resp.status == 200
        finally:
            server.shutdown()

    def test_metrics_returns_prometheus_format(self):
        trigger = threading.Event()
        port = _pick_port()
        server = ragstuffer._start_admin_server(trigger, port, "")
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
            data = resp.read()
            assert b"ragstuffer" in data
        finally:
            server.shutdown()

    def test_metrics_content_type_is_plain_text(self):
        trigger = threading.Event()
        port = _pick_port()
        server = ragstuffer._start_admin_server(trigger, port, "")
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
            assert "text/plain" in resp.headers.get("Content-Type", "")
        finally:
            server.shutdown()

    def test_health_still_works(self):
        trigger = threading.Event()
        port = _pick_port()
        server = ragstuffer._start_admin_server(trigger, port, "")
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            assert resp.status == 200
            assert json.loads(resp.read()) == {"status": "ok"}
        finally:
            server.shutdown()

    def test_metrics_no_auth_required(self):
        trigger = threading.Event()
        port = _pick_port()
        server = ragstuffer._start_admin_server(trigger, port, "secret-token")
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
            assert resp.status == 200
        finally:
            server.shutdown()
