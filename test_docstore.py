"""Tests for docstore.py — SQLite backend + CachedDocstore wrapper.

Tests use SQLite (no Postgres dependency needed for CI).
"""

import os

import pytest

# Force SQLite backend before importing
os.environ["DOCSTORE_BACKEND"] = "sqlite"

from docstore import CachedDocstore, SQLiteDocstore, create_docstore


@pytest.fixture
def sqlite_store(tmp_path):
    """Fresh SQLite docstore for each test."""
    db_path = str(tmp_path / "test.db")
    store = SQLiteDocstore(db_path)
    store.init_schema()
    return store


@pytest.fixture
def cached_store(sqlite_store):
    """CachedDocstore wrapping a fresh SQLite backend."""
    return CachedDocstore(sqlite_store, maxsize=64)


# ── SQLiteDocstore ───────────────────────────────────────────────────────────


class TestSQLiteDocstore:
    def test_init_schema_idempotent(self, sqlite_store):
        # Calling init_schema twice should not raise
        sqlite_store.init_schema()

    def test_upsert_and_get_chunk(self, sqlite_store):
        sqlite_store.upsert_chunk("doc1", 0, "Hello world", "test.md")
        result = sqlite_store.get_chunk("doc1", 0)
        assert result == "Hello world"

    def test_get_missing_chunk(self, sqlite_store):
        assert sqlite_store.get_chunk("missing", 0) is None

    def test_upsert_overwrites(self, sqlite_store):
        sqlite_store.upsert_chunk("doc1", 0, "v1", "test.md")
        sqlite_store.upsert_chunk("doc1", 0, "v2", "test.md")
        assert sqlite_store.get_chunk("doc1", 0) == "v2"

    def test_upsert_chunks_batch(self, sqlite_store):
        chunks = [
            {"doc_id": "doc1", "chunk_id": 0, "text": "Chunk 0", "source": "a.md"},
            {"doc_id": "doc1", "chunk_id": 1, "text": "Chunk 1", "source": "a.md"},
            {"doc_id": "doc2", "chunk_id": 0, "text": "Chunk 2", "source": "b.md"},
        ]
        sqlite_store.upsert_chunks(chunks)
        assert sqlite_store.get_chunk("doc1", 0) == "Chunk 0"
        assert sqlite_store.get_chunk("doc1", 1) == "Chunk 1"
        assert sqlite_store.get_chunk("doc2", 0) == "Chunk 2"

    def test_get_chunks_batch(self, sqlite_store):
        sqlite_store.upsert_chunk("d1", 0, "A", "s")
        sqlite_store.upsert_chunk("d1", 1, "B", "s")
        sqlite_store.upsert_chunk("d2", 0, "C", "s")
        result = sqlite_store.get_chunks([("d1", 0), ("d1", 1), ("d2", 0)])
        assert result == {("d1", 0): "A", ("d1", 1): "B", ("d2", 0): "C"}

    def test_get_chunks_empty_refs(self, sqlite_store):
        assert sqlite_store.get_chunks([]) == {}

    def test_delete_doc(self, sqlite_store):
        sqlite_store.upsert_chunk("doc1", 0, "A", "s")
        sqlite_store.upsert_chunk("doc1", 1, "B", "s")
        sqlite_store.upsert_chunk("doc2", 0, "C", "s")
        sqlite_store.delete_doc("doc1")
        assert sqlite_store.get_chunk("doc1", 0) is None
        assert sqlite_store.get_chunk("doc1", 1) is None
        assert sqlite_store.get_chunk("doc2", 0) == "C"


# ── CachedDocstore ───────────────────────────────────────────────────────────


class TestCachedDocstore:
    def test_cache_hit(self, cached_store):
        cached_store.upsert_chunk("d1", 0, "cached", "s")
        # First get populates cache, second is a hit
        cached_store.get_chunk("d1", 0)
        cached_store.get_chunk("d1", 0)
        stats = cached_store.cache_stats
        assert stats["hits"] >= 1

    def test_cache_miss(self, cached_store):
        cached_store.get_chunk("missing", 0)
        assert cached_store.cache_stats["misses"] >= 1

    def test_upsert_invalidates_cache(self, cached_store):
        cached_store.upsert_chunk("d1", 0, "v1", "s")
        assert cached_store.get_chunk("d1", 0) == "v1"
        cached_store.upsert_chunk("d1", 0, "v2", "s")
        assert cached_store.get_chunk("d1", 0) == "v2"

    def test_delete_evicts_cache(self, cached_store):
        cached_store.upsert_chunk("d1", 0, "A", "s")
        cached_store.upsert_chunk("d1", 1, "B", "s")
        cached_store.get_chunk("d1", 0)  # populate cache
        cached_store.delete_doc("d1")
        assert cached_store.get_chunk("d1", 0) is None

    def test_batch_upsert_populates_cache(self, cached_store):
        chunks = [
            {"doc_id": "d1", "chunk_id": 0, "text": "A", "source": "s"},
            {"doc_id": "d1", "chunk_id": 1, "text": "B", "source": "s"},
        ]
        cached_store.upsert_chunks(chunks)
        # These should be cache hits
        assert cached_store.get_chunk("d1", 0) == "A"
        assert cached_store.get_chunk("d1", 1) == "B"
        assert cached_store.cache_stats["hits"] >= 2

    def test_batch_get_uses_cache(self, cached_store):
        cached_store.upsert_chunk("d1", 0, "A", "s")
        cached_store.upsert_chunk("d1", 1, "B", "s")
        # Batch get should use cache for already-cached entries
        result = cached_store.get_chunks([("d1", 0), ("d1", 1)])
        assert result == {("d1", 0): "A", ("d1", 1): "B"}

    def test_cache_eviction_on_maxsize(self):
        """Cache evicts oldest entries when maxsize is exceeded."""
        store = SQLiteDocstore(":memory:")
        store.init_schema()
        cached = CachedDocstore(store, maxsize=3)
        for i in range(5):
            cached.upsert_chunk("d", i, f"text-{i}", "s")
        assert cached.cache_stats["size"] == 3

    def test_init_schema_delegates(self, cached_store):
        # Should not raise
        cached_store.init_schema()


# ── create_docstore factory ──────────────────────────────────────────────────


class TestCreateDocstore:
    def test_sqlite_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DOCSTORE_SQLITE_PATH", str(tmp_path / "factory.db"))
        store = create_docstore(backend="sqlite")
        assert isinstance(store, CachedDocstore)
        store.upsert_chunk("d1", 0, "test", "s")
        assert store.get_chunk("d1", 0) == "test"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown DOCSTORE_BACKEND"):
            create_docstore(backend="redis")

    def test_postgres_without_url_raises(self, monkeypatch):
        monkeypatch.setenv("DOCSTORE_URL", "")
        with pytest.raises(ValueError, match="DOCSTORE_URL must be set"):
            create_docstore(backend="postgres", url="")
