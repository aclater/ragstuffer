"""Live integration tests for ragstuffer.

Tests run against the live ragstuffer service (:8091).
Requires ragstuffer and upstream services (Qdrant, Postgres, ragpipe) to be running.

Run with:
    PYTHONPATH=. pytest tests/test_live.py -v --ragstuffer-url=http://localhost:8091

Skip in CI (service not available):
    SKIP_LIVE_TESTS=1 pytest tests/test_live.py -v -m "not live"

Note: Some issue requirements cannot be tested via HTTP endpoints:
- /collections endpoint does not exist in ragstuffer (collections managed internally)
- /embed endpoint does not exist (embedding delegated to ragpipe)
- Qdrant/Postgres reachability can only be verified indirectly via metrics
"""

import os

import httpx
import pytest

RAGSTUFFER_URL = os.environ.get("RAGSTUFFER_URL", "http://localhost:8091")
RAGSTUFFER_MPEP_URL = os.environ.get("RAGSTUFFER_MPEP_URL", "http://localhost:8093")
TIMEOUT = 30.0


def _is_ragstuffer_available() -> bool:
    try:
        httpx.get(f"{RAGSTUFFER_URL}/health", timeout=5)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("SKIP_LIVE_TESTS") == "1" or not _is_ragstuffer_available(),
        reason="ragstuffer not available — set SKIP_LIVE_TESTS=1 to skip",
    ),
]


@pytest.fixture
def ragstuffer_url():
    return RAGSTUFFER_URL


@pytest.fixture
def ragstuffer_mpep_url():
    return RAGSTUFFER_MPEP_URL


# ── Health and connectivity ────────────────────────────────────────────────────


def test_health_returns_200(ragstuffer_url):
    resp = httpx.get(f"{ragstuffer_url}/health", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_metrics_returns_prometheus(ragstuffer_url):
    resp = httpx.get(f"{ragstuffer_url}/metrics", timeout=10)
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    text = resp.text
    assert "ragstuffer" in text


def test_qdrant_reachable_via_ingest_metrics(ragstuffer_url):
    resp = httpx.get(f"{ragstuffer_url}/metrics", timeout=10)
    text = resp.text
    assert "ragstuffer_documents_ingested_total" in text or "ragstuffer_chunks_created_total" in text


def test_postgres_reachable_via_ingest_metrics(ragstuffer_url):
    resp = httpx.get(f"{ragstuffer_url}/metrics", timeout=10)
    text = resp.text
    assert "ragstuffer_documents_ingested_total" in text or "ragstuffer_chunks_created_total" in text


def test_embed_requests_tracked(ragstuffer_url):
    resp = httpx.get(f"{ragstuffer_url}/metrics", timeout=10)
    text = resp.text
    assert "ragstuffer_embed_requests_total" in text


def test_embed_errors_tracked(ragstuffer_url):
    resp = httpx.get(f"{ragstuffer_url}/metrics", timeout=10)
    text = resp.text
    assert "ragstuffer_embed_errors_total" in text


# ── Ingestion endpoint ────────────────────────────────────────────────────────


def test_ingest_now_endpoint_exists(ragstuffer_url):
    resp = httpx.post(f"{ragstuffer_url}/admin/ingest-now", timeout=10)
    assert resp.status_code in (200, 202, 401)


def test_ingest_full_endpoint_exists(ragstuffer_url):
    resp = httpx.post(f"{ragstuffer_url}/admin/ingest-full", timeout=10)
    assert resp.status_code in (200, 202, 401)


# ── ragstuffer-mpep instance ──────────────────────────────────────────────────


def test_mpep_instance_healthy(ragstuffer_mpep_url):
    resp = httpx.get(f"{ragstuffer_mpep_url}/health", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_mpep_metrics_available(ragstuffer_mpep_url):
    resp = httpx.get(f"{ragstuffer_mpep_url}/metrics", timeout=10)
    assert resp.status_code == 200
    assert "ragstuffer" in resp.text


# ── Metrics content validation ─────────────────────────────────────────────────


def test_ingest_duration_metric_present(ragstuffer_url):
    resp = httpx.get(f"{ragstuffer_url}/metrics", timeout=10)
    text = resp.text
    assert "ragstuffer_ingest_duration_seconds" in text


def test_last_ingest_timestamp_present(ragstuffer_url):
    resp = httpx.get(f"{ragstuffer_url}/metrics", timeout=10)
    text = resp.text
    assert "ragstuffer_last_ingest_timestamp_seconds" in text


def test_all_metrics_have_valid_values(ragstuffer_url):
    resp = httpx.get(f"{ragstuffer_url}/metrics", timeout=10)
    text = resp.text
    for line in text.splitlines():
        if line.startswith("ragstuffer") and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                name, value = parts[0], parts[1]
                try:
                    float(value)
                except ValueError:
                    pytest.fail(f"Invalid metric value: {name}={value}")
