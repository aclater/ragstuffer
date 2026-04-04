#!/usr/bin/env python3
"""One-shot GPU ingestion — runs on a remote host with sentence-transformers.

Auto-detects GPU (NVIDIA CUDA, AMD ROCm, Intel XPU) for embedding,
falls back to CPU. Pushes chunks to Postgres and vectors to Qdrant
on a target host.

Usage:
    # Via deploy script:
    ./deploy-remote.sh <gpu-host> [target-host]

    # Or manually:
    TARGET_HOST=<target-ip> GDRIVE_FOLDER_ID=<id> python3 ingest-remote.py
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from common import (
    ALL_EXTENSIONS,
    EXPORT_MAP,
    GDRIVE_SCOPES,
    _extract_html_title,
    chunk_text,
    extract_html_text,
    extract_text_with_title,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ingest-remote")

# ── Configuration ────────────────────────────────────────────────────────────

TARGET_HOST_IP = os.environ.get("TARGET_HOST", "127.0.0.1")
QDRANT_URL = os.environ.get("QDRANT_URL", f"http://{TARGET_HOST_IP}:6333")
DOCSTORE_URL = os.environ.get("DOCSTORE_URL", f"postgresql://litellm:litellm@{TARGET_HOST_IP}:5432/litellm")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "documents")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# CHUNK_SIZE/CHUNK_OVERLAP are passed to common.chunk_text()
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "128"))

# Google Drive
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "")
GDRIVE_SA_KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", str(Path.home() / ".config/ramalama/gdrive-sa.json"))

# Git repos: JSON list of {"url": "...", "glob": "**/*.md"}
REPO_SOURCES = os.environ.get("REPO_SOURCES", "")

# Web URLs: JSON list of URLs
WEB_SOURCES = os.environ.get("WEB_SOURCES", "")


# ── Google Drive ─────────────────────────────────────────────────────────────


def load_drive_docs(staging_dir: Path) -> list[dict]:
    if not GDRIVE_FOLDER_ID:
        log.info("Drive: no GDRIVE_FOLDER_ID set, skipping")
        return []

    sa_path = Path(GDRIVE_SA_KEY).expanduser()
    if not sa_path.exists():
        log.warning("Drive: SA key not found at %s, skipping", sa_path)
        return []

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(sa_path)

    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    creds = service_account.Credentials.from_service_account_file(str(sa_path), scopes=GDRIVE_SCOPES)
    service = build("drive", "v3", credentials=creds)

    # List all files in folder
    files = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{GDRIVE_FOLDER_ID}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageSize=100,
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    log.info("Drive: found %d files", len(files))

    downloaded: list[Path] = []
    for f in files:
        file_id = f["id"]
        name = f["name"]
        mime = f["mimeType"]

        if mime in EXPORT_MAP:
            export_mime, ext = EXPORT_MAP[mime]
            dest = staging_dir / f"{Path(name).stem}{ext}"
            request = service.files().export_media(fileId=file_id, mimeType=export_mime)
        else:
            ext = Path(name).suffix.lower()
            if ext not in ALL_EXTENSIONS:
                log.debug("Skipping unsupported: %s", name)
                continue
            dest = staging_dir / name
            request = service.files().get_media(fileId=file_id)

        log.info("Downloading %s", name)
        with open(dest, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        downloaded.append(dest)

    docs = []
    for f in downloaded:
        if not f.is_file():
            continue
        extracted = extract_text_with_title(f)
        if extracted.text:
            docs.append({"text": extracted.text, "source": f"gdrive://{f.name}", "title": extracted.title})
        else:
            log.warning("Drive: no text from %s", f.name)
        f.unlink(missing_ok=True)

    return docs


# ── Git source ───────────────────────────────────────────────────────────────


def load_git_docs(repos_dir: Path) -> list[dict]:
    if not REPO_SOURCES:
        return []

    try:
        sources = json.loads(REPO_SOURCES)
    except json.JSONDecodeError:
        log.error("REPO_SOURCES is not valid JSON")
        return []

    docs = []
    for source in sources:
        url = source.get("url", "")
        glob_pattern = source.get("glob", "**/*")
        if not url:
            continue

        repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_path = repos_dir / repo_name

        try:
            if repo_path.exists():
                log.info("Git: pulling %s", repo_name)
                result = subprocess.run(
                    ["git", "-C", str(repo_path), "pull", "--ff-only"],
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    shutil.rmtree(repo_path)
                    subprocess.run(
                        ["git", "clone", "--depth", "1", url, str(repo_path)],
                        check=True,
                        capture_output=True,
                        timeout=120,
                    )
            else:
                log.info("Git: cloning %s (shallow)", repo_name)
                subprocess.run(
                    ["git", "clone", "--depth", "1", url, str(repo_path)],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
        except Exception:
            log.warning("Git: failed to clone/pull %s, skipping", url)
            continue

        for file_path in repo_path.glob(glob_pattern):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in ALL_EXTENSIONS:
                continue
            try:
                rel = file_path.relative_to(repo_path)
                extracted = extract_text_with_title(file_path)
                if extracted.text:
                    docs.append({"text": extracted.text, "source": f"{url}@{rel}", "title": extracted.title})
            except Exception:
                log.warning("Git: failed to extract %s — skipping", file_path)

    log.info("Git: loaded %d documents from %d repos", len(docs), len(sources))
    return docs


# ── Web source ───────────────────────────────────────────────────────────────


def load_web_docs() -> list[dict]:
    if not WEB_SOURCES:
        return []

    try:
        urls = json.loads(WEB_SOURCES)
    except json.JSONDecodeError:
        log.error("WEB_SOURCES is not valid JSON")
        return []

    import requests

    docs = []
    for url in urls:
        try:
            log.info("Web: fetching %s", url)
            resp = requests.get(url, timeout=30, headers={"User-Agent": "ragstuffer/1.0"})
            resp.raise_for_status()
            if "html" in resp.headers.get("content-type", "").lower():
                text = extract_html_text(resp.text)
                title = _extract_html_title(resp.text)
            else:
                text = resp.text
                title = ""
            if text.strip():
                docs.append({"text": text, "source": url, "title": title})
            else:
                log.warning("Web: no text extracted from %s", url)
        except Exception:
            log.warning("Web: failed to fetch %s — skipping", url)

    log.info("Web: loaded %d documents from %d URLs", len(docs), len(urls))
    return docs


# ── Local FastEmbed ──────────────────────────────────────────────────────────


def _detect_device() -> str:
    """Auto-detect the best available accelerator for PyTorch.

    Priority: CUDA (NVIDIA) > ROCm (AMD via HIP) > XPU (Intel) > CPU.
    ROCm exposes AMD GPUs as CUDA devices via HIP, so torch.cuda covers
    both NVIDIA and AMD when the ROCm PyTorch build is installed.
    Override with RAGSTUFFER_DEVICE env var (cuda, xpu, cpu).
    """
    import torch

    forced = os.environ.get("RAGSTUFFER_DEVICE", "").strip().lower()
    if forced:
        log.info("Device forced via RAGSTUFFER_DEVICE=%s", forced)
        return forced

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        log.info("Detected GPU: %s (CUDA/ROCm)", name)
        return "cuda"

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        log.info("Detected Intel XPU")
        return "xpu"

    log.info("No GPU detected — using CPU")
    return "cpu"


def embed_texts_local(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    """Embed using sentence-transformers on the best available accelerator."""
    from sentence_transformers import SentenceTransformer

    device = _detect_device()
    log.info("Loading model %s on %s...", EMBED_MODEL, device)
    model = SentenceTransformer(EMBED_MODEL, device=device)

    log.info("Embedding %d texts (batch_size=%d, device=%s)...", len(texts), batch_size, device)
    t0 = time.monotonic()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.monotonic() - t0
    log.info(
        "Embedded %d texts in %.1fs (%.0f texts/sec)",
        len(texts),
        elapsed,
        len(texts) / elapsed if elapsed > 0 else 0,
    )
    return vectors.tolist()


# ── Qdrant + docstore ───────────────────────────────────────────────────────


def _point_id(doc_id: str, chunk_id: int) -> int:
    h = hashlib.sha256(f"{doc_id}:{chunk_id}".encode()).hexdigest()
    return int(h[:16], 16)


def ingest(docs: list[dict]) -> None:
    import psycopg2
    from psycopg2.extras import execute_values
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
        VectorParams,
    )

    if not docs:
        log.info("No documents to ingest")
        return

    # Chunk all documents
    all_chunks = []
    source_types: set[str] = set()
    for doc in docs:
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc["source"]))
        chunks = chunk_text(doc["text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        title = doc.get("title", "")
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "text": chunk,
                    "source": doc["source"],
                    "title": title,
                }
            )
        source_types.add(_classify_source_type(doc["source"]))
        log.info("Chunked %s -> %d chunks (doc_id=%s)", doc["source"], len(chunks), doc_id)

    if not all_chunks:
        log.warning("No text chunks produced")
        return

    log.info("Total: %d chunks from %d documents", len(all_chunks), len(docs))

    # Step 1: Embed locally with FastEmbed
    texts = [c["text"] for c in all_chunks]
    vectors = embed_texts_local(texts)

    # Step 2: Push chunks to Postgres docstore on target host
    log.info("Connecting to Postgres at %s...", TARGET_HOST_IP)
    conn = psycopg2.connect(DOCSTORE_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                doc_id     TEXT NOT NULL,
                chunk_id   INTEGER NOT NULL,
                text       TEXT NOT NULL,
                source     TEXT NOT NULL DEFAULT '',
                title      TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (doc_id, chunk_id)
            )
        """)
        now = datetime.now(UTC).isoformat()
        values = [(c["doc_id"], c["chunk_id"], c["text"], c["source"], c.get("title", ""), now) for c in all_chunks]
        execute_values(
            cur,
            """
            INSERT INTO chunks (doc_id, chunk_id, text, source, title, created_at)
            VALUES %s
            ON CONFLICT (doc_id, chunk_id)
            DO UPDATE SET text = EXCLUDED.text, source = EXCLUDED.source, title = EXCLUDED.title
            """,
            values,
        )
    conn.close()
    log.info("Persisted %d chunks to Postgres", len(all_chunks))

    # Register collection in Postgres (non-fatal if table missing)
    _register_collection(QDRANT_COLLECTION, source_types)

    # Step 3: Upsert vectors to Qdrant on target host
    log.info("Connecting to Qdrant at %s...", QDRANT_URL)
    qdrant = QdrantClient(url=QDRANT_URL, timeout=60)

    collections = [c.name for c in qdrant.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            ),
        )
        log.info("Created Qdrant collection '%s'", QDRANT_COLLECTION)

    now = datetime.now(UTC).isoformat()
    points = [
        PointStruct(
            id=_point_id(chunk["doc_id"], chunk["chunk_id"]),
            vector=vec,
            payload={
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "created_at": now,
            },
        )
        for vec, chunk in zip(vectors, all_chunks, strict=True)
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points[i : i + batch_size])
        if (i // batch_size) % 10 == 0:
            log.info("Qdrant upsert: %d/%d points", min(i + batch_size, len(points)), len(points))

    log.info("Ingested %d points into Qdrant collection '%s'", len(points), QDRANT_COLLECTION)


def _classify_source_type(source: str) -> str:
    """Classify a document source string into a source type category."""
    if source.startswith("gdrive://"):
        return "drive"
    if source.startswith("http://") or source.startswith("https://"):
        if "@" in source.split("//", 1)[1]:
            return "git"
        return "web"
    return "git"


def _register_collection(collection_name: str, source_types: set[str]) -> None:
    """Upsert a row in the collections table on first ingest.

    If the collections table does not exist (migrations not run), logs a
    clear warning and skips — ingestion continues normally.
    """
    from datetime import datetime

    import psycopg2

    try:
        conn = psycopg2.connect(DOCSTORE_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            now = datetime.now(UTC).isoformat()
            source_types_json = json.dumps(sorted(source_types))
            cur.execute(
                """
                INSERT INTO collections (name, source_types, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (name)
                DO UPDATE SET
                    source_types = COALESCE(
                        (SELECT jsonb_agg(DISTINCT elem ORDER BY elem)
                         FROM (
                             SELECT jsonb_array_elements_text(collections.source_types::jsonb) AS elem
                             UNION
                             SELECT jsonb_array_elements_text(%s::jsonb) AS elem
                         ) sub
                        ),
                        %s
                    ),
                    updated_at = %s
                """,
                (collection_name, source_types_json, source_types_json, source_types_json, now),
            )
        conn.close()
        log.info("Registered collection '%s' with source types %s", collection_name, source_types)
    except Exception as e:
        err_msg = str(e)
        if "relation \"collections\" does not exist" in err_msg or "no such table" in err_msg:
            log.warning(
                "collections table not found — skipping collection registration. "
                "Run rag-suite migrations first."
            )
        else:
            log.warning("Failed to register collection '%s': %s", collection_name, err_msg)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    staging_dir = Path("/tmp/rag-staging")
    repos_dir = Path("/tmp/rag-repos")
    staging_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)

    log.info("One-shot ingestion — embedding locally, pushing to %s", TARGET_HOST_IP)
    log.info("  Qdrant:     %s", QDRANT_URL)
    log.info("  Postgres:   %s@%s", "litellm", TARGET_HOST_IP)
    log.info("  Collection: %s", QDRANT_COLLECTION)
    log.info("  Model:      %s", EMBED_MODEL)

    all_docs = []
    all_docs.extend(load_drive_docs(staging_dir))
    all_docs.extend(load_git_docs(repos_dir))
    all_docs.extend(load_web_docs())

    if not all_docs:
        log.info("No documents found from any source")
        return

    log.info("Total: %d documents from all sources", len(all_docs))

    t0 = time.monotonic()
    ingest(all_docs)
    elapsed = time.monotonic() - t0
    log.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    main()
