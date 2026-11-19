# SimpleRAG — Model & RAG Workbench

A single-user FastAPI web prototype: OpenAI-compatible model health checks and chat, BGE embeddings, MinerU/MarkItDown document parsing and batch conversion, Milvus vector search, MinIO object storage, and a SQLite token usage ledger.

> Current testing-stage version: `v0.1.0`. Chinese documentation: [README.md](README.md).

## Features

| Page | Description |
| --- | --- |
| Overview | Pick and save the default chat / embedding models; health of models and Milvus / MinIO / MinerU; usage stats |
| Knowledge Base | Upload → store original → split → embed → write to Milvus, with six visible stages; re-index and delete documents |
| Conversion | MinerU batch conversion into a unified JSON (`schema: rag.document`) plus Markdown; afterwards choose "continue to chunk & index" or "download ZIP" |
| Chat | Optionally enable RAG retrieval; answers cite their sources |
| Token Usage | Per-model call counts and tokens, plus a recent-call table |
| Storage Settings | Edit the MinIO / Milvus host and port at runtime, and jump to each service's management console |

## Storage settings and auto-provisioning

The **Storage Settings** page lets you change the Milvus (gRPC, default `19530`) and MinIO (S3 API, default `9000`) host and port. Changes take effect immediately and are stored in the local SQLite database (`app_settings` table) — **`.env` is never rewritten**. "Restore .env defaults" clears the overrides. The page also provides direct links to:

- Milvus console: `http://<host>:9091`
- MinIO console: `http://<host>:9001`

Tolerance:

- The bucket named by `MINIO_BUCKET` is created when it does not exist.
- The Milvus database named by `MILVUS_DATABASE` is created when it does not exist. Milvus only allows letters, digits and underscores in database names; illegal characters are replaced with `_` and a warning is logged.
- A missing collection is created with the actual embedding dimension, an AUTOINDEX index, and is loaded.
- "Test connection" returns an error within seconds when a host is unreachable instead of hanging.

## Configuration

```bash
cp .env.example .env      # then fill in AI_API_KEY and the service endpoints
```

`.env` holds secrets — **never commit it to any repository or send it to any external service**. Only `.env.example` belongs in git.

## Running

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -r requirements.txt
.venv/bin/python app.py
```

Open `http://127.0.0.1:8000` (host and port come from `HOST` / `PORT` in `.env`); the API docs are at `/docs`.

## Development notes

All modules import as `agent_test.rag.*`. To run scripts or tests locally, put a symlink to this repository inside a directory on `PYTHONPATH` rather than rewriting the imports:

```bash
mkdir -p /tmp/rag_shim/agent_test
touch /tmp/rag_shim/agent_test/__init__.py
ln -s "$PWD" /tmp/rag_shim/agent_test/rag
PYTHONPATH=/tmp/rag_shim .venv/bin/python -c "import agent_test.rag.app"
```

Do not add `-I`: isolated mode ignores `PYTHONPATH`.

## Limitations

- Token counts come from the API's `usage` field; calls without usage are marked estimated/missing.
- PDFs and images go through MinerU and fall back to MarkItDown when it is unconfigured or unreachable (weaker PDF extraction).
- RAG refuses to query with an embedding model that does not match the stored vectors; switching models requires re-indexing.
- Conversions run on an in-process thread pool; unfinished jobs are marked failed/retryable after a restart.
- Request logs record only path, status code and latency — never keys, prompt bodies or document content.
