# ragstuffer

Document ingestion for RAG pipelines. Polls Google Drive, git repos, and web URLs, extracts text, chunks, embeds, and stuffs everything into Qdrant + Postgres for retrieval by [ragpipe](https://github.com/aclater/ragpipe).

## What it does

1. **Poll** document sources on a configurable interval (default 15 minutes)
2. **Download** new/modified documents (Google Drive via service account, git shallow clone, web fetch)
3. **Extract** text from PDF, DOCX, PPTX, XLSX, HTML, Markdown, plain text — with **title extraction per source type**
4. **Chunk** with RecursiveCharacterTextSplitter (1024 chars, 128 overlap)
5. **Persist** chunks to Postgres document store (keyed on deterministic UUID5 from source URI), including title metadata per source
6. **Embed** via ragpipe's `/v1/embeddings` endpoint (or directly via sentence-transformers with GPU auto-detection)
7. **Upsert** reference-only payloads to Qdrant (vectors + {doc_id, chunk_id, source, title, created_at})

## Title extraction

Titles are extracted per source type and stored alongside chunk metadata in Postgres, then surfaced by ragpipe in `rag_metadata.cited_chunks[].title`:

| Source type | Title extraction |
|-------------|------------------|
| Google Drive (PDF) | PDF metadata `Title` field, or filename without extension |
| Google Drive (DOCX/PPTX) | Document title from Office metadata, or filename |
| Google Drive (XLSX) | Sheet name or filename |
| Google Drive (other) | Filename |
| git repos | First Markdown heading (`# Title`) in file, or filename |
| Web URLs | `<title>` tag content, or URL path |
| Local files | Filename |

Title extraction enables ragpipe to surface document titles in citations without storing full text in Qdrant.

## Multiple collection support

ragstuffer can ingest into multiple Qdrant collections simultaneously. Collections are registered in the `collections` table in Postgres, and each source can target a specific collection via the `COLLECTION` environment variable:

| Environment variable | Description |
|---------------------|-------------|
| `QDRANT_COLLECTION` | Single collection name (backward compatible) |
| `QDRANT_COLLECTIONS` | JSON array of collection names for multi-collection ingest |

The `collections` table tracks collection metadata:

```sql
SELECT * FROM collections;
-- id | name | source_type | description | created_at
-- 1  | personnel | gdrive | Personnel documents | 2024-01-01
-- 2  | nato | git | NATO documents | 2024-01-02
```

## Second instance (ragstuffer-mpep)

A second ragstuffer instance (`ragstuffer-mpep`) runs alongside the primary instance to ingest the USPTO/MPEP patent collection into the `mpep` Qdrant collection. It runs on port 8093:

| Instance | Port | Collection | Description |
|----------|------|------------|-------------|
| ragstuffer | 8091 | personnel, nato, documents | General document ingestion |
| ragstuffer-mpep | 8093 | mpep | USPTO/MPEP patent collection |

Both instances share the same Postgres database but write to different Qdrant collections.

## Quick start (container)

Three Containerfile variants — build selects automatically based on GPU:

| Containerfile | Base image | GPU | Use case |
|---------------|-----------|-----|----------|
| `Containerfile` | UBI10 (Red Hat) | none | CPU-only poller, delegates embedding to ragpipe |
| `Containerfile.rocm` | rocm/pytorch (AMD) | ROCm | GPU embedding on AMD GPUs |
| `Containerfile.cuda` | pytorch/pytorch (NVIDIA) | CUDA | GPU embedding on NVIDIA GPUs |

```bash
# Tag convention: localhost/ragstuffer:main (or :$(git rev-parse --short HEAD))
# CPU-only (default — Red Hat UBI10)
podman build -t localhost/ragstuffer:main .

# AMD ROCm GPU
podman build -t localhost/ragstuffer:rocm -f Containerfile.rocm .

# NVIDIA CUDA GPU
podman build -t localhost/ragstuffer:cuda -f Containerfile.cuda .
```

Or let `llm-stack.sh build` auto-select based on detected GPU.

```bash
podman run --rm \
    -e GDRIVE_FOLDER_ID=your-folder-id \
    -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gdrive-sa.json \
    -e QDRANT_URL=http://host.containers.internal:6333 \
    -e EMBED_URL=http://host.containers.internal:8090/v1/embeddings \
    -e DOCSTORE_URL=postgresql://user:pass@host.containers.internal:5432/db \
    -v ~/.config/ramalama/gdrive-sa.json:/run/secrets/gdrive-sa.json:ro \
    ragstuffer
```

## One-shot GPU ingestion

For bulk ingestion, `ingest-remote.py` auto-detects the best available GPU (NVIDIA CUDA, AMD ROCm, Intel XPU) and embeds using sentence-transformers, then pushes results to Qdrant + Postgres over the network. Falls back to CPU if no GPU is found.

```bash
# Deploy to a remote GPU host, pushing data to a target host:
GDRIVE_FOLDER_ID=your-folder-id ./deploy-remote.sh gpu-host.local target-host.local

# Or embed and store on the same remote host:
GDRIVE_FOLDER_ID=your-folder-id ./deploy-remote.sh gpu-host.local
```

## Configuration

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `GDRIVE_FOLDER_ID` | — | Google Drive folder to watch |
| `REPO_SOURCES` | — | JSON list: `[{"url": "...", "glob": "**/*.md"}]` |
| `WEB_SOURCES` | — | JSON list: `["https://..."]` |
| `WATCH_INTERVAL_MINUTES` | `15` | Poll interval |
| `CHUNK_SIZE` | `1024` | Max chunk size (characters) |
| `CHUNK_OVERLAP` | `128` | Overlap between chunks |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_URL` | `http://127.0.0.1:8090/v1/embeddings` | Embedding endpoint (ragpipe) |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name |
| `QDRANT_COLLECTIONS` | — | JSON array of collection names for multi-collection ingest |
| `DOCSTORE_BACKEND` | `postgres` | `postgres` or `sqlite` |
| `DOCSTORE_URL` | *(required)* | Postgres connection string |

### Google Drive

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Path to service account key |

### Admin

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGSTUFFER_ADMIN_PORT` | `8091` | Admin API listen port |
| `RAGSTUFFER_ADMIN_TOKEN` | *(none)* | Bearer token for admin endpoints (set to secure access) |

### GPU ingestion (`ingest-remote.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_HOST` | `127.0.0.1` | Hostname/IP of Qdrant + Postgres target |
| `RAGSTUFFER_DEVICE` | *(auto-detect)* | Force device: `cuda`, `xpu`, or `cpu` |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence-transformers model |

GPU auto-detection priority: CUDA (NVIDIA) > ROCm (AMD via HIP) > XPU (Intel) > CPU.

## Admin API

Ragstuffer runs a lightweight admin server for triggering ingestion without waiting for the poll interval and exposing metrics.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/ingest-now` | POST | Trigger immediate incremental ingestion |
| `/admin/ingest-full` | POST | Clear state file + trigger full re-ingest of all documents |
| `/health` | GET | Liveness check |
| `/metrics` | GET | Prometheus metrics |

```bash
# Trigger immediate ingestion
curl -X POST http://localhost:8091/admin/ingest-now -H "Authorization: Bearer <token>"

# Force full re-ingest (re-downloads everything)
curl -X POST http://localhost:8091/admin/ingest-full -H "Authorization: Bearer <token>"
```

## Prometheus metrics

```
# HELP ragstuffer_documents_ingested_total Documents ingested from all sources
# TYPE ragstuffer_documents_ingested_total counter
ragstuffer_documents_ingested_total{source="gdrive"} 150
ragstuffer_documents_ingested_total{source="git"} 42
ragstuffer_documents_ingested_total{source="web"} 8

# HELP ragstuffer_chunks_created_total Chunks created from ingested documents
# TYPE ragstuffer_chunks_created_total counter
ragstuffer_chunks_created_total 3420

# HELP ragstuffer_embed_requests_total Embedding requests sent to ragpipe
# TYPE ragstuffer_embed_requests_total counter
ragstuffer_embed_requests_total 200

# HELP ragstuffer_embed_errors_total Embedding request failures
# TYPE ragstuffer_embed_errors_total counter
ragstuffer_embed_errors_total 3
```

## Project structure

```
ragstuffer/
  common.py           — shared constants, text extraction, chunking, title extraction, HTML parsing
  docstore.py         — Postgres/SQLite backends + LRU-cached docstore wrapper
  ragstuffer/
    __init__.py      — package marker
    metrics.py        — Prometheus metrics definitions
  ragstuffer          — main poll loop, admin server, graceful shutdown (executable)
  ingest-remote.py    — one-shot GPU ingestion (sentence-transformers)
  setup.sh            — interactive setup wizard (SA key, folder ID, quadlet)
  deploy-remote.sh    — deploy ingest-remote.py to a GPU host via ssh
  quadlets/           — Podman quadlet for systemd integration
  Containerfile       — UBI10 CPU-only image
  Containerfile.rocm  — AMD ROCm GPU image
  Containerfile.cuda  — NVIDIA CUDA GPU image
```

## Running tests

```bash
pip install -r requirements.txt
python -m pytest -v    # 100 tests
```

## License

AGPL-3.0-or-later
