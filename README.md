# ragstuffer

Document ingestion for RAG pipelines. Polls Google Drive, git repos, and web URLs, extracts text, chunks, embeds, and stuffs everything into Qdrant + Postgres for retrieval by [ragpipe](https://github.com/aclater/ragpipe).

## Table of contents

- [Architecture](docs/architecture.md) — data flow, key design decisions, project structure
- [Configuration](docs/configuration.md) — environment variables
- [Admin API](docs/admin-api.md) — endpoints and Prometheus metrics
- [Title extraction](docs/title-extraction.md) — how titles are extracted per source type

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

## Running tests

```bash
pip install -r requirements.txt
python -m pytest -v    # 100+ tests
```

## License

AGPL-3.0-or-later
