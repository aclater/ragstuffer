# Configuration

## Core

| Variable | Default | Description |
|----------|---------|-------------|
| `GDRIVE_FOLDER_ID` | — | Google Drive folder to watch |
| `REPO_SOURCES` | — | JSON list: `[{"url": "...", "glob": "**/*.md"}]` |
| `WEB_SOURCES` | — | JSON list: `["https://..."]` |
| `WATCH_INTERVAL_MINUTES` | `15` | Poll interval |
| `CHUNK_SIZE` | `1024` | Max chunk size (characters) |
| `CHUNK_OVERLAP` | `128` | Overlap between chunks |

## Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_URL` | `http://127.0.0.1:8090/v1/embeddings` | Embedding endpoint (ragpipe) |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name |
| `QDRANT_COLLECTIONS` | — | JSON array of collection names for multi-collection ingest |
| `DOCSTORE_BACKEND` | `postgres` | `postgres` or `sqlite` |
| `DOCSTORE_URL` | *(required)* | Postgres connection string |

## Google Drive

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Path to service account key |

## Admin

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGSTUFFER_ADMIN_PORT` | `8091` | Admin API listen port |
| `RAGSTUFFER_ADMIN_TOKEN` | *(none)* | Bearer token for admin endpoints (set to secure access) |

## GPU ingestion (`ingest-remote.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_HOST` | `127.0.0.1` | Hostname/IP of Qdrant + Postgres target |
| `RAGSTUFFER_DEVICE` | *(auto-detect)* | Force device: `cuda`, `xpu`, or `cpu` |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence-transformers model |

GPU auto-detection priority: CUDA (NVIDIA) > ROCm (AMD via HIP) > XPU (Intel) > CPU.
