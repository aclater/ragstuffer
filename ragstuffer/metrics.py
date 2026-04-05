from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

ragstuffer_documents_ingested_total = Counter(
    "ragstuffer_documents_ingested_total",
    "Total documents ingested, tagged by source type and collection",
    ["source_type", "collection"],
    registry=REGISTRY,
)

ragstuffer_chunks_created_total = Counter(
    "ragstuffer_chunks_created_total",
    "Total chunks created during ingestion",
    ["collection"],
    registry=REGISTRY,
)

ragstuffer_ingest_duration_seconds = Histogram(
    "ragstuffer_ingest_duration_seconds",
    "Duration of ingest_docs() call in seconds",
    ["source_type"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
    registry=REGISTRY,
)

ragstuffer_embed_requests_total = Counter(
    "ragstuffer_embed_requests_total",
    "Total embed batch requests sent to ragpipe",
    registry=REGISTRY,
)

ragstuffer_embed_errors_total = Counter(
    "ragstuffer_embed_errors_total",
    "Total embed request errors",
    registry=REGISTRY,
)

ragstuffer_last_ingest_timestamp = Gauge(
    "ragstuffer_last_ingest_timestamp_seconds",
    "Unix timestamp of last successful ingest completion",
    ["collection"],
    registry=REGISTRY,
)

ragstuffer_documents_skipped_total = Counter(
    "ragstuffer_documents_skipped_total",
    "Total documents skipped",
    ["reason"],
    registry=REGISTRY,
)


def get_metrics() -> bytes:
    return generate_latest(REGISTRY)
