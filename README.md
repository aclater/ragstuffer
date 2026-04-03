# ragstuffer

Document ingestion for RAG pipelines. Polls Google Drive, git repos, and web URLs, extracts text, chunks, embeds, and stuffs everything into Qdrant + Postgres for retrieval by [ragpipe](https://github.com/aclater/ragpipe).

## What it does

1. **Poll** document sources on a configurable interval (default 15 minutes)
2. **Download** new/modified documents (Google Drive via service account, git shallow clone, web fetch)
3. **Extract** text from PDF, DOCX, PPTX, XLSX, HTML, Markdown, plain text
4. **Chunk** with RecursiveCharacterTextSplitter (1024 chars, 128 overlap)
5. **Persist** chunks to Postgres document store (keyed on deterministic UUID5 from source URI)
6. **Embed** via ragpipe's `/v1/embeddings` endpoint (or directly via sentence-transformers for GPU)
7. **Upsert** reference-only payloads to Qdrant (vectors + {doc_id, chunk_id, source, created_at})

## Quick start (container)

```bash
podman build -t ragstuffer .
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

For bulk ingestion, `ingest-remote.py` runs embedding on a GPU-equipped host using sentence-transformers (CUDA) and pushes results to Qdrant + Postgres over the network:

```bash
# On a host with an NVIDIA GPU:
./deploy-to-lennon.sh
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GDRIVE_FOLDER_ID` | — | Google Drive folder to watch |
| `REPO_SOURCES` | — | JSON list: `[{"url": "...", "glob": "**/*.md"}]` |
| `WEB_SOURCES` | — | JSON list: `["https://..."]` |
| `WATCH_INTERVAL_MINUTES` | `15` | Poll interval |
| `CHUNK_SIZE` | `1024` | Max chunk size (characters) |
| `CHUNK_OVERLAP` | `128` | Overlap between chunks |
| `EMBED_URL` | `http://127.0.0.1:8090/v1/embeddings` | Embedding endpoint (ragpipe) |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name |
| `DOCSTORE_BACKEND` | `postgres` | `postgres` or `sqlite` |
| `DOCSTORE_URL` | *(required)* | Postgres connection string |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Path to service account key |

## Running tests

```bash
pip install -r requirements.txt
python -m pytest test_ragstuffer.py -v
```

## License

AGPL-3.0-or-later
