#!/bin/bash
# Deploy and run one-shot RAG ingestion on a remote GPU host.
# Embeds locally on the remote host, pushes to a target Qdrant + Postgres.
#
# Usage: ./deploy-remote.sh <gpu-host> [target-host]
#   gpu-host:    hostname/IP of the machine with GPU (runs embedding)
#   target-host: hostname/IP where Qdrant + Postgres are (receives data)
#                defaults to gpu-host if not specified (embed + store on same host)
#
# Environment:
#   GDRIVE_FOLDER_ID  — required, Google Drive folder to ingest
#   SA_KEY            — path to service account key (default: ~/.config/ramalama/gdrive-sa.json)
#   REPO_SOURCES      — optional JSON list of git repos
#   WEB_SOURCES       — optional JSON list of web URLs
set -euo pipefail

REMOTE="${1:?Usage: $0 <gpu-host> [target-host]}"
TARGET="${2:-$REMOTE}"
REMOTE_DIR="/tmp/rag-ingest"
SA_KEY="${SA_KEY:-$HOME/.config/ramalama/gdrive-sa.json}"

echo "=== Deploying to $REMOTE (target: $TARGET) ==="

# Create remote directory and copy files
ssh "$REMOTE" "mkdir -p $REMOTE_DIR"
scp ingest-remote.py "$REMOTE:$REMOTE_DIR/"

# Copy SA key if it exists
if [[ -f "$SA_KEY" ]]; then
    ssh "$REMOTE" "mkdir -p ~/.config/ramalama"
    scp "$SA_KEY" "$REMOTE:~/.config/ramalama/gdrive-sa.json"
    echo "Copied SA key"
fi

echo "=== Installing dependencies ==="
ssh "$REMOTE" "cd $REMOTE_DIR && \
    python3 -m venv --system-site-packages venv && \
    source venv/bin/activate && \
    pip install -q \
        sentence-transformers \
        qdrant-client \
        psycopg2-binary \
        langchain-text-splitters \
        google-api-python-client \
        google-auth-httplib2 \
        google-auth-oauthlib \
        pypdf \
        python-docx \
        python-pptx \
        openpyxl \
        requests"

echo "=== Running ingestion ==="
ssh "$REMOTE" "cd $REMOTE_DIR && \
    source venv/bin/activate && \
    HARRISON_HOST=$TARGET \
    GDRIVE_FOLDER_ID=${GDRIVE_FOLDER_ID:?Set GDRIVE_FOLDER_ID before running} \
    REPO_SOURCES='${REPO_SOURCES:-}' \
    WEB_SOURCES='${WEB_SOURCES:-}' \
    EMBED_THREADS=\$(nproc) \
    python3 ingest-remote.py"
