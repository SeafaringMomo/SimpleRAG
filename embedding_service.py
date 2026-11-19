"""Chunking and vector persistence for BGE backed document embedding."""
from __future__ import annotations

import hashlib
import re
import logging
from agent_test.rag.config import settings
from agent_test.rag.model_registry import registry

logger = logging.getLogger(__name__)


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    size = chunk_size if chunk_size is not None else settings.chunk_size
    requested_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if size < 100 or size > 20000:
        raise ValueError("chunk_size must be between 100 and 20000 characters")
    if requested_overlap < 0 or requested_overlap >= size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    overlap = min(requested_overlap, size // 2)
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind("。", start, end), text.rfind(" ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [c for c in chunks if c]


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_collection(client, dim: int, collection_name: str | None = None) -> bool:
    collection_name = collection_name or settings.collection
    from pymilvus import DataType
    created = False
    if not client.has_collection(collection_name):
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("doc_name", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        client.create_collection(collection_name=collection_name, schema=schema)
        created = True
        logger.info("Milvus collection created name=%s dimension=%d", collection_name, dim)
    else:
        description = client.describe_collection(collection_name)
        vector_field = next((f for f in description.get("fields", []) if f.get("name") == "vector"), None)
        existing_dim = (vector_field or {}).get("params", {}).get("dim")
        if existing_dim is not None and int(existing_dim) != dim:
            raise ValueError(f"Collection {collection_name} dimension is {existing_dim}; selected embedding model returned {dim}")
    vector_indexes = client.list_indexes(collection_name, field_name="vector")
    index_created = False
    if not vector_indexes:
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_index(collection_name=collection_name, index_params=index_params)
        index_created = True
        logger.info("Milvus vector index created name=%s dimension=%d created_collection=%s",
                    collection_name, dim, created)
    client.load_collection(collection_name)
    logger.info("Milvus collection loaded name=%s", collection_name)
    return index_created


def insert_chunks(doc_id: str, doc_name: str, chunks: list[str], replace: bool = False,
                  embedding_model: str | None = None, collection_name: str | None = None,
                  vectors: list[list[float]] | None = None) -> int:
    if not chunks:
        return 0
    collection_name = collection_name or settings.collection
    vectors = vectors if vectors is not None else registry.embed(chunks, embedding_model)
    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise ValueError("Embedding model returned inconsistent vector dimensions")
    dim = dims.pop()
    if settings.embedding_dimension and settings.embedding_dimension != dim:
        raise ValueError(f"Configured dimension {settings.embedding_dimension} differs from model dimension {dim}")
    from agent_test.rag.storage import milvus_client
    client = milvus_client()
    ensure_collection(client, dim, collection_name)
    rows = [{"id": f"{doc_id}-{i}", "doc_id": doc_id, "doc_name": doc_name,
             "chunk_index": i, "content": chunk, "vector": vector}
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))]
    if replace:
        # Upsert current indexes, then remove stale trailing indexes.
        client.upsert(collection_name=collection_name, data=rows)
        client.delete(collection_name=collection_name,
                      filter=f'doc_id == "{doc_id}" and chunk_index >= {len(rows)}')
    else:
        client.insert(collection_name=collection_name, data=rows)
    logger.info("Milvus vectors persisted collection=%s doc_id=%s rows=%d dimension=%d replace=%s",
                collection_name, doc_id, len(rows), dim, replace)
    return len(rows)


def delete_document_vectors(doc_id: str, collection_name: str | None = None) -> None:
    from agent_test.rag.storage import milvus_client
    client = milvus_client()
    collection_name = collection_name or settings.collection
    if client.has_collection(collection_name):
        description = client.describe_collection(collection_name)
        vector_field = next((f for f in description.get("fields", []) if f.get("name") == "vector"), None)
        dim = int((vector_field or {}).get("params", {}).get("dim", 0))
        if dim <= 0:
            raise ValueError(f"Collection {collection_name} has no valid vector field")
        ensure_collection(client, dim, collection_name)
        client.delete(collection_name=collection_name, filter=f'doc_id == "{doc_id}"')
        logger.info("Milvus vectors deleted collection=%s doc_id=%s", collection_name, doc_id)
