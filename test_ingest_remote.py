"""Tests for ingest-remote.py — Qdrant payload consistency with ragstuffer."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Real PointStruct stand-in ───────────────────────────────────────────────
# We need a real class (not MagicMock) so payload dict access works correctly.


@dataclass
class _PointStruct:
    id: int
    vector: list
    payload: dict


# Stub heavy dependencies not available in test environment
_STUBS = {}
for _mod in [
    "google.oauth2.service_account",
    "googleapiclient.discovery",
    "googleapiclient.http",
    "sentence_transformers",
    "psycopg2",
    "psycopg2.extras",
]:
    if _mod not in sys.modules:
        _STUBS[_mod] = MagicMock()
        sys.modules[_mod] = _STUBS[_mod]

# qdrant_client stub with real PointStruct
_qc_mod = sys.modules.get("qdrant_client")
if _qc_mod is None or isinstance(_qc_mod, MagicMock):
    _qc_mod = MagicMock()
    sys.modules["qdrant_client"] = _qc_mod
_qc_models = sys.modules.get("qdrant_client.models")
if _qc_models is None or isinstance(_qc_models, MagicMock):
    _qc_models = MagicMock()
    sys.modules["qdrant_client.models"] = _qc_models
# Always inject the real PointStruct so payload dicts are inspectable
_qc_models.PointStruct = _PointStruct

# Ensure langchain_text_splitters is real (used by common.chunk_text)
_lts = sys.modules.get("langchain_text_splitters")
if _lts is not None and isinstance(_lts, MagicMock):
    del sys.modules["langchain_text_splitters"]

spec = importlib.util.spec_from_file_location(
    "ingest_remote",
    Path(__file__).with_name("ingest-remote.py"),
)
ir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir)


def _run_ingest_and_capture(docs, fake_vectors):
    """Run ir.ingest() with mocked dependencies and return captured Qdrant points."""
    mock_qdrant_client = MagicMock()
    mock_qdrant_client.get_collections.return_value.collections = []

    captured_points = []

    def _capture_upsert(**kwargs):
        captured_points.extend(kwargs.get("points", []))

    mock_qdrant_client.upsert.side_effect = _capture_upsert
    _qc_mod.QdrantClient = MagicMock(return_value=mock_qdrant_client)

    with (
        patch.object(ir, "embed_texts_local", return_value=fake_vectors),
        patch("psycopg2.connect", return_value=MagicMock()),
    ):
        ir.ingest(docs)

    return captured_points


class TestQdrantPayloadIncludesTitle:
    """Verify ingest-remote Qdrant payload matches ragstuffer (includes title)."""

    def test_qdrant_points_have_title_field(self):
        """The Qdrant PointStruct payload must include 'title' for parity with ragstuffer."""
        docs = [{"text": "Hello world content.", "source": "gdrive://test.txt", "title": "Test Doc"}]
        points = _run_ingest_and_capture(docs, [[0.1, 0.2, 0.3]])

        assert len(points) > 0, "No points captured from upsert"
        for point in points:
            assert "title" in point.payload, f"Qdrant payload missing 'title': {point.payload}"
            assert point.payload["title"] == "Test Doc"

    def test_title_empty_string_when_not_provided(self):
        """Documents without a title key should get empty string, not KeyError."""
        docs = [{"text": "No title doc.", "source": "gdrive://notitle.txt"}]
        points = _run_ingest_and_capture(docs, [[0.1, 0.2, 0.3]])

        assert len(points) > 0
        for point in points:
            assert "title" in point.payload
            assert point.payload["title"] == ""
