# ragstuffer

Document ingestion for RAG pipelines. Polls Google Drive, git repos, and web URLs, extracts text, chunks, embeds, and stuffs everything into Qdrant + Postgres for retrieval by ragpipe.

## Architecture
```
Document sources (Google Drive / git / web)
        ↓ poll + download
Text extraction (PDF, DOCX, PPTX, XLSX, HTML, Markdown, plain text)
        ↓ title extraction per source type
Chunking (RecursiveCharacterTextSplitter, 1024 chars, 128 overlap)
        ↓
Embed via ragpipe /v1/embeddings (or sentence-transformers for ingest-remote.py)
        ↓
Upsert to Qdrant (vectors + {doc_id, chunk_id, source, title, created_at})
        ↓
Persist to Postgres (chunks + titles, keyed by deterministic UUID5 from source URI)
```

## Title extraction

Titles are extracted per source type and stored alongside chunk metadata in Postgres:

| Source type | Title source |
|-------------|-------------|
| PDF | PDF metadata Title, or filename without extension |
| DOCX/PPTX | Office document title, or filename |
| XLSX | Sheet name or filename |
| git/Markdown | First `# Heading` in file, or filename |
| Web URLs | `<title>` tag, or URL path |
| Local files | Filename |

Titles are surfaced by ragpipe in `rag_metadata.cited_chunks[].title`.

## Package structure
```
ragstuffer/
  common.py           — shared constants, text extraction, chunking, title extraction, HTML parsing
  docstore.py         — Postgres/SQLite backends + LRU-cached docstore wrapper
  ragstuffer/
    __init__.py      — package marker
    metrics.py       — Prometheus metrics definitions
  ragstuffer          — main poll loop, admin server, graceful shutdown (executable)
  ingest-remote.py    — one-shot GPU ingestion (sentence-transformers)
  setup.sh            — interactive setup wizard (SA key, folder ID, quadlet)
  deploy-remote.sh    — deploy ingest-remote.py to a GPU host via ssh
  quadlets/           — Podman quadlet for systemd integration
  Containerfile       — UBI10 CPU-only image
  Containerfile.rocm  — AMD ROCm GPU image
  Containerfile.cuda  — NVIDIA CUDA GPU image
```

## Key design decisions
- Deterministic UUID5 keys from source URI — re-ingest is idempotent
- Incremental updates — only changed documents are re-downloaded
- Title extraction per source type — enables ragpipe to surface document titles in citations
- Embedding delegated to ragpipe (`/v1/embeddings`) in polling mode — no GPU needed for ingestion
- `ingest-remote.py` for bulk GPU-accelerated embedding (sentence-transformers with auto GPU detection)
- Multiple collection support via `QDRANT_COLLECTIONS` JSON env var
- Collections registered in `collections` table in Postgres

## Multiple collections

ragstuffer can ingest into multiple Qdrant collections. The `collections` table
tracks metadata:

```sql
CREATE TABLE collections (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    source_type TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

Set `QDRANT_COLLECTION` for single collection (backward compatible) or
`QDRANT_COLLECTIONS='["personnel", "nato", "mpep"]'` for multi-collection.

## GPU auto-detection (ingest-remote.py)

Priority: CUDA (NVIDIA) > ROCm (AMD via HIP) > XPU (Intel) > CPU.

```python
import torch
if torch.cuda.is_available():
    device = "cuda"
elif torch.version.hip:
    device = "cuda"  # AMD ROCm uses CUDA device in sentence-transformers
elif torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"
```

## Admin endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/ingest-now` | POST | Trigger immediate incremental ingestion |
| `/admin/ingest-full` | POST | Clear state file + full re-ingest |
| `/health` | GET | Liveness check |
| `/metrics` | GET | Prometheus metrics |

## Prometheus metrics

```
ragstuffer_documents_ingested_total{source="gdrive|git|web"}
ragstuffer_chunks_created_total
ragstuffer_embed_requests_total
ragstuffer_embed_errors_total
```

## Running tests
```bash
pip install -r requirements.txt
python -m pytest -v    # 100 tests
```

## Container images

Three variants built by `llm-stack.sh build`:
- CPU: UBI10, delegates embedding to ragpipe
- ROCm: rocm/pytorch, AMD GPU embedding
- CUDA: pytorch/pytorch, NVIDIA GPU embedding
