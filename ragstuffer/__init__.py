#!/usr/bin/env python3
"""ragstuffer — ingests documents from Google Drive, git repos, and web URLs into Qdrant.

Document loading patterns (git shallow clone, glob filtering, web extraction, chunking
with source attribution) are adapted from the Red Hat Validated Patterns vector-embedder:
https://github.com/validatedpatterns-sandbox/vector-embedder
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from common import (  # noqa: E402,F401
    ALL_EXTENSIONS,
    EXPORT_MAP,
    GDRIVE_SCOPES,
    _extract_html_title,
    chunk_text,
    extract_html_text,
    extract_text,
    extract_text_with_title,
)
from ragstuffer.metrics import get_metrics  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ragstuffer")

# ── Configuration ────────────────────────────────────────────────────────────

STATE_PATH = Path(
    os.environ.get(
        "RAG_STATE_FILE",
        Path.home() / ".local" / "share" / "ramalama" / "rag-state.json",
    )
)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


# ── Google Drive source ──────────────────────────────────────────────────────


def get_drive_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_file:
        return None

    creds_path = Path(creds_file).expanduser()
    if not creds_path.exists():
        log.warning("Service account key not found: %s — Drive source disabled", creds_path)
        return None

    creds = service_account.Credentials.from_service_account_file(str(creds_path), scopes=GDRIVE_SCOPES)
    return build("drive", "v3", credentials=creds)


def list_drive_files(service, folder_id: str) -> list[dict]:
    results = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageSize=100,
                pageToken=page_token,
            )
            .execute()
        )
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def download_drive_file(
    service, file_info: dict, staging_dir: Path, *, max_retries: int = 3
) -> Path | None:
    """Download a Drive file to staging_dir. Returns the dest Path on success, None on skip.

    Retries transient HTTP errors (403 rate-limit, 429, 500, 503) with
    exponential backoff. Google Docs export is limited to 10 MB by the API;
    files exceeding this limit are logged and skipped.
    """
    import time

    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload

    file_id = file_info["id"]
    name = file_info["name"]
    mime = file_info["mimeType"]

    if mime in EXPORT_MAP:
        export_mime, ext = EXPORT_MAP[mime]
        dest = staging_dir / f"{Path(name).stem}{ext}"
        log.info("Exporting %s → %s", name, dest.name)
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        ext = Path(name).suffix.lower()
        if ext not in ALL_EXTENSIONS:
            log.debug("Skipping unsupported file: %s", name)
            return None
        dest = staging_dir / name
        log.info("Downloading %s", name)
        request = service.files().get_media(fileId=file_id)

    _RETRYABLE_STATUS_CODES = {403, 429, 500, 503}

    for attempt in range(max_retries):
        try:
            with open(dest, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            return dest
        except HttpError as e:
            status = e.resp.status if e.resp else 0
            if status == 403 and "exportSizeLimitExceeded" in str(e):
                log.warning(
                    "Drive: %s exceeds 10 MB export limit — skipping (Google Docs "
                    "export is capped at 10 MB by the API)",
                    name,
                )
                dest.unlink(missing_ok=True)
                return None
            if status in _RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                backoff = 2**attempt
                log.warning(
                    "Drive: transient HTTP %d downloading %s — retrying in %ds (attempt %d/%d)",
                    status,
                    name,
                    backoff,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(backoff)
                continue
            raise
    return None


def load_drive_docs(staging_dir: Path) -> list[dict]:
    """Load Drive documents: return list of {text, source, title} dicts."""
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    if not folder_id:
        return []

    service = get_drive_service()
    if not service:
        return []

    state = load_state()
    files = list_drive_files(service, folder_id)
    changed = [f for f in files if state.get(f["id"]) != f["modifiedTime"]]

    if not changed:
        log.info("Drive: no new or modified files")
        return []

    log.info("Drive: %d new/modified file(s)", len(changed))
    # Prune state entries for files deleted from Drive
    current_ids = {f["id"] for f in files}
    new_state = {fid: mtime for fid, mtime in state.items() if fid in current_ids}

    downloaded: list[Path] = []
    for f in changed:
        try:
            dest = download_drive_file(service, f, staging_dir)
            if dest is not None:
                new_state[f["id"]] = f["modifiedTime"]
                downloaded.append(dest)
        except Exception:
            log.warning("Drive: failed to download %s — skipping", f["name"], exc_info=True)

    save_state(new_state)

    docs = []
    for f in downloaded:
        if not f.is_file():
            continue
        extracted = extract_text_with_title(f)
        if extracted.text:
            docs.append({"text": extracted.text, "source": f"gdrive://{f.name}", "title": extracted.title})
        else:
            log.warning("Drive: no text extracted from %s", f.name)
        f.unlink(missing_ok=True)
    return docs


# ── Git source (adapted from vector-embedder's GitLoader) ───────────────────


def load_git_docs(repos_dir: Path) -> list[dict]:
    """Clone/pull git repos and extract text from matching files.

    REPO_SOURCES is a JSON list of objects: [{"url": "...", "glob": "**/*.md"}, ...]
    Shallow clones and incremental pull adapted from vector-embedder.
    """
    sources_json = os.environ.get("REPO_SOURCES", "")
    if not sources_json:
        return []

    try:
        sources = json.loads(sources_json)
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
                    log.warning("Git: pull failed for %s, re-cloning", repo_name)
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
            log.warning("Git: failed to clone/pull %s — skipping", url)
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


# ── Web source (adapted from vector-embedder's WebLoader) ───────────────────


def load_web_docs() -> list[dict]:
    """Fetch web pages and extract text.

    WEB_SOURCES is a JSON list of URLs: ["https://example.com/docs", ...]
    """
    sources_json = os.environ.get("WEB_SOURCES", "")
    if not sources_json:
        return []

    try:
        urls = json.loads(sources_json)
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


# ── Qdrant ingestion ─────────────────────────────────────────────────────────


def get_qdrant_client():
    from qdrant_client import QdrantClient

    url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    return QdrantClient(url=url, timeout=30)


def _wait_for_embed_service(embed_url: str, timeout: int = 300) -> None:
    """Block until the embedding service health check passes.

    Ragpipe's /health returns 503 while MIGraphX is compiling the ONNX
    graph (~3 min on gfx1151). Polling here avoids wasting a full ingest
    cycle on a connection refused that will resolve on its own.
    """
    import time

    import requests

    health_url = embed_url.rsplit("/v1/", 1)[0] + "/health"
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                if attempt > 0:
                    log.info("Embedding service ready after %d attempts", attempt)
                return
        except (requests.ConnectionError, requests.Timeout):
            pass
        attempt += 1
        backoff = min(10, 2 ** min(attempt, 5))
        log.info("Embedding service not ready (attempt %d), retrying in %ds...", attempt, backoff)
        time.sleep(backoff)
    raise RuntimeError(f"Embedding service at {health_url} not ready after {timeout}s")


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed texts via ragpipe's /v1/embeddings endpoint.

    Delegates to ragpipe instead of loading a local model — saves ~1-2 GB RAM.
    Sends batches in parallel (up to 4 concurrent) with connection pooling.
    Output order is preserved via batch indexing.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests

    from ragstuffer.metrics import ragstuffer_embed_errors_total, ragstuffer_embed_requests_total

    embed_url = os.environ.get("EMBED_URL", "http://127.0.0.1:8090/v1/embeddings")

    _wait_for_embed_service(embed_url)

    session = requests.Session()

    def _embed_batch(batch_idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
        ragstuffer_embed_requests_total.inc()
        try:
            resp = session.post(embed_url, json={"input": batch}, timeout=120)
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda x: x["index"])
            return batch_idx, [d["embedding"] for d in data]
        except Exception:
            ragstuffer_embed_errors_total.inc()
            raise

    batches = [(i, texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    results: list[list[list[float]] | None] = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_embed_batch, idx, batch): idx for idx, (_, batch) in enumerate(batches)}
        for future in as_completed(futures):
            batch_idx, vectors = future.result()
            results[batch_idx] = vectors

    return [v for batch in results for v in batch]


def ensure_collection(qdrant, collection_name: str, vector_size: int):
    """Create Qdrant collection with scalar int8 quantization.

    Quantization reduces memory footprint of vectors. always_ram=True is
    required because HNSW rescoring needs quantized vectors in RAM for
    accurate distance computation during search.

    NOTE: Qdrant does not support adding quantization to an existing
    collection in place. If the collection already exists without
    quantization, it must be dropped and recreated.
    """
    from qdrant_client.models import Distance, ScalarQuantization, ScalarQuantizationConfig, ScalarType, VectorParams

    collections = [c.name for c in qdrant.get_collections().collections]
    if collection_name not in collections:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            ),
        )
        log.info("Created Qdrant collection '%s' with int8 scalar quantization", collection_name)


def _point_id(doc_id: str, chunk_id: int) -> int:
    """Deterministic point ID from doc_id:chunk_id for idempotent Qdrant upsert."""
    h = hashlib.sha256(f"{doc_id}:{chunk_id}".encode()).hexdigest()
    return int(h[:16], 16)


def _get_docstore():
    """Lazy-import and create docstore to avoid import at module level."""
    from docstore import create_docstore

    return create_docstore()


def _get_existing_doc_ids(qdrant, collection_name: str) -> set[str]:
    """Return the set of doc_ids that already have vectors in Qdrant."""

    try:
        collections = [c.name for c in qdrant.get_collections().collections]
        if collection_name not in collections:
            return set()

        doc_ids: set[str] = set()
        offset = None
        while True:
            results = qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=1000,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            points, next_offset = results
            for point in points:
                if doc_id := point.payload.get("doc_id"):
                    doc_ids.add(doc_id)
            if next_offset is None:
                break
            offset = next_offset

        return doc_ids
    except Exception:
        log.warning("Failed to query Qdrant for existing doc_ids — will re-embed all")
        return set()


def ingest_docs(docs: list[dict]) -> bool:
    """Chunk documents, persist to docstore, embed, and upsert references to Qdrant.

    Skips embedding for documents that already have vectors in Qdrant
    (checked by doc_id). Postgres upsert always runs (cheap, idempotent).
    Qdrant payloads contain only {doc_id, chunk_id, source, title, created_at}.
    Full chunk text lives in the document store.
    """
    import time
    from datetime import datetime

    from qdrant_client.models import PointStruct

    from ragstuffer.metrics import (
        ragstuffer_chunks_created_total,
        ragstuffer_documents_ingested_total,
        ragstuffer_documents_skipped_total,
        ragstuffer_ingest_duration_seconds,
        ragstuffer_last_ingest_timestamp,
    )

    if not docs:
        log.info("No documents to ingest")
        return True

    collection_name = os.environ.get("QDRANT_COLLECTION", "documents")
    chunk_size = int(os.environ.get("CHUNK_SIZE", "1024"))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "128"))

    all_chunks = []
    source_types: set[str] = set()
    for doc in docs:
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc["source"]))
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
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
        st = _classify_source_type(doc["source"])
        source_types.add(st)
        log.info("Chunked %s → %d chunks (doc_id=%s)", doc["source"], len(chunks), doc_id)

    if not all_chunks:
        log.warning("No text chunks to ingest")
        return True

    docstore = _get_docstore()
    docstore.upsert_chunks(all_chunks)
    log.info("Persisted %d chunks to docstore", len(all_chunks))

    _register_collection(collection_name, source_types)

    qdrant = get_qdrant_client()
    existing_doc_ids = _get_existing_doc_ids(qdrant, collection_name)

    new_chunks = [c for c in all_chunks if c["doc_id"] not in existing_doc_ids]
    skipped = len(all_chunks) - len(new_chunks)
    if skipped:
        log.info(
            "Skipping %d chunks (%d docs) already in Qdrant",
            skipped,
            len({c["doc_id"] for c in all_chunks} - {c["doc_id"] for c in new_chunks}),
        )
        ragstuffer_documents_skipped_total.labels(reason="already_indexed").inc(skipped)

    if not new_chunks:
        log.info("All %d chunks already in Qdrant — nothing to embed", len(all_chunks))
        return True

    start_time = time.monotonic()

    embed_batch_size = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
    log.info("Embedding %d new chunks via ragpipe (batch_size=%d)...", len(new_chunks), embed_batch_size)
    texts = [c["text"] for c in new_chunks]
    vectors = embed_texts(texts, batch_size=embed_batch_size)

    ensure_collection(qdrant, collection_name, len(vectors[0]))

    now = datetime.now(UTC).isoformat()
    points = [
        PointStruct(
            id=_point_id(chunk["doc_id"], chunk["chunk_id"]),
            vector=vec,
            payload={
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "title": chunk["title"],
                "created_at": now,
            },
        )
        for vec, chunk in zip(vectors, new_chunks, strict=True)
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name=collection_name, points=points[i : i + batch_size])

    duration = time.monotonic() - start_time
    primary_source = next(iter(source_types), "unknown")
    ragstuffer_ingest_duration_seconds.labels(source_type=primary_source).observe(duration)
    ragstuffer_documents_ingested_total.labels(source_type=primary_source, collection=collection_name).inc(len(docs))
    ragstuffer_chunks_created_total.labels(collection=collection_name).inc(len(new_chunks))
    ragstuffer_last_ingest_timestamp.labels(collection=collection_name).set(time.time())

    log.info("Ingested %d reference points into Qdrant collection '%s'", len(points), collection_name)
    return True


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

    try:
        docstore = _get_docstore()
        backend = docstore._backend
        if not hasattr(backend, "_get_sync_conn"):
            return
        conn = backend._get_sync_conn()
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
                (collection_name, source_types_json, now, source_types_json, source_types_json, now),
            )
        log.info("Registered collection '%s' with source types %s", collection_name, source_types)
    except Exception as e:
        err_msg = str(e)
        if 'relation "collections" does not exist' in err_msg or "no such table" in err_msg:
            log.warning(
                "collections table not found — skipping collection registration. Run rag-suite migrations first."
            )
        else:
            log.warning("Failed to register collection '%s': %s", collection_name, err_msg)


# ── Poll loop ────────────────────────────────────────────────────────────────


def poll_once(staging_dir: Path, repos_dir: Path) -> None:
    """Single poll iteration: gather docs from all sources, ingest into Qdrant."""
    from concurrent.futures import ThreadPoolExecutor

    all_docs = []

    # Load from all sources in parallel — they are independent I/O operations
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(load_drive_docs, staging_dir),
            executor.submit(load_git_docs, repos_dir),
            executor.submit(load_web_docs),
        ]
        for future in futures:
            all_docs.extend(future.result())

    if not all_docs:
        log.info("No documents to process from any source")
        return

    log.info("Total: %d documents from all sources", len(all_docs))

    if ingest_docs(all_docs):
        log.info("Qdrant updated — new documents available for RAG immediately")
    else:
        log.warning("Ingestion failed — will retry next poll")


def _start_admin_server(trigger_event, admin_port, admin_token):
    """Run a minimal HTTP admin server for triggering ingestion.

    Endpoints:
      POST /admin/ingest-now   — trigger immediate poll (incremental)
      POST /admin/ingest-full  — clear state + trigger full re-ingest
      GET  /health             — liveness check
    """
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class AdminHandler(BaseHTTPRequestHandler):
        def _check_auth(self):
            if not admin_token:
                return True
            auth = self.headers.get("Authorization", "")
            if auth == f"Bearer {admin_token}":
                return True
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"unauthorized"}')
            return False

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            elif self.path == "/metrics":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(get_metrics())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if not self._check_auth():
                return

            if self.path == "/admin/ingest-now":
                log.info("Admin trigger: ingest-now (incremental)")
                trigger_event.set()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_json.dumps({"status": "triggered", "mode": "incremental"}).encode())

            elif self.path == "/admin/ingest-full":
                log.info("Admin trigger: ingest-full (clearing state)")
                if STATE_PATH.exists():
                    STATE_PATH.unlink()
                    log.info("Cleared state file: %s", STATE_PATH)
                trigger_event.set()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_json.dumps({"status": "triggered", "mode": "full"}).encode())

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            """Suppress default access logging — we log triggers explicitly."""

    server = HTTPServer(("0.0.0.0", admin_port), AdminHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info("Admin server listening on :%d", admin_port)
    return server


def main() -> None:
    import signal
    import threading

    interval = int(os.environ.get("WATCH_INTERVAL_MINUTES", "15"))
    admin_port = int(os.environ.get("RAGSTUFFER_ADMIN_PORT", "8091"))
    admin_token = os.environ.get("RAGSTUFFER_ADMIN_TOKEN", "")
    staging_dir = Path(os.environ.get("STAGING_DIR", "/tmp/rag-staging"))
    repos_dir = Path(os.environ.get("REPOS_DIR", "/tmp/rag-repos"))

    staging_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Event for admin-triggered immediate ingestion
    trigger = threading.Event()
    # Event for graceful shutdown
    shutdown = threading.Event()

    def _handle_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        log.info("Received %s — shutting down gracefully", sig_name)
        shutdown.set()
        trigger.set()  # unblock the wait

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    log.info("Starting ragstuffer")
    log.info("  Qdrant:     %s", os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"))
    log.info("  Collection: %s", os.environ.get("QDRANT_COLLECTION", "documents"))
    log.info("  Interval:   %d min", interval)
    log.info("  Admin:      :%d", admin_port)
    if os.environ.get("GDRIVE_FOLDER_ID"):
        log.info("  Drive:      folder %s", os.environ["GDRIVE_FOLDER_ID"])
    if os.environ.get("REPO_SOURCES"):
        log.info("  Git:        %s", os.environ["REPO_SOURCES"])
    if os.environ.get("WEB_SOURCES"):
        log.info("  Web:        %s", os.environ["WEB_SOURCES"])

    server = _start_admin_server(trigger, admin_port, admin_token)

    while not shutdown.is_set():
        try:
            poll_once(staging_dir, repos_dir)
        except Exception:
            log.exception("Poll cycle failed")
        # Wait for interval OR admin trigger, whichever comes first
        trigger.wait(timeout=interval * 60)
        trigger.clear()

    server.shutdown()
    log.info("Shutdown complete")


if __name__ == "__main__":
    main()
