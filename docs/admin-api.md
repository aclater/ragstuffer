# Admin API

Ragstuffer runs a lightweight admin server for triggering ingestion without waiting for the poll interval and exposing metrics.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/ingest-now` | POST | Trigger immediate incremental ingestion |
| `/admin/ingest-full` | POST | Clear state file + trigger full re-ingest of all documents |
| `/health` | GET | Liveness check |
| `/metrics` | GET | Prometheus metrics |

## Usage

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
