"""Document conversion and indexing pipeline."""
from __future__ import annotations

import io
import os
import uuid
import json
import logging
import zipfile
from datetime import datetime, timezone
from agent_test.rag.config import settings
from agent_test.rag.embedding_service import content_hash, delete_document_vectors, ensure_collection, insert_chunks, split_text
from agent_test.rag.object_store import put_file, remove_file
from agent_test.rag.usage_store import connection, update_ingest_task

logger = logging.getLogger(__name__)

# Plain text handled locally; no network service and no MarkItDown dependency.
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".log"}


def parse_document(path: str, filename: str, raw: bytes | None = None,
                   progress=None) -> dict:
    """Convert any supported file into the unified document JSON.

    Routing order: plain text → MinerU (when configured for the extension) →
    MarkItDown → plain text fallback. MinerU failures fall back to a local
    converter only when `MINERU_FALLBACK_TO_MARKITDOWN` is enabled; the
    resulting `source.converter` records which path was actually used.
    `progress(task)` is forwarded to the MinerU client for queue reporting.
    """
    from agent_test.rag import document_parser, mineru_client
    ext = os.path.splitext(filename)[1].lower()

    if ext in TEXT_EXTENSIONS:
        return document_parser.parse_text_file(path, filename, raw)

    if mineru_client.should_use_mineru(filename):
        try:
            return mineru_client.parse_with_mineru(path, filename, raw, progress=progress)
        except mineru_client.MinerUError as exc:
            if not settings.mineru_fallback:
                raise
            logger.warning("MinerU unavailable, falling back to MarkItDown filename=%s error=%s",
                           filename, exc)
    try:
        return document_parser.parse_with_markitdown(path, filename, raw)
    except Exception as exc:
        logger.warning("MarkItDown failed filename=%s error=%s", filename, exc)
        if ext in {".txt", ".md", ".markdown", ".csv"}:
            return document_parser.parse_text_file(path, filename, raw)
        raise


def convert_document(path: str, filename: str, raw: bytes | None = None) -> str:
    """Backwards-compatible helper returning only the flattened text."""
    from agent_test.rag import document_parser
    document = parse_document(path, filename, raw)
    text = document_parser.document_text(document)
    if not text:
        raise ValueError("No text was extracted from the uploaded document")
    return text


def conversion_object_keys(doc_id: str, filename: str) -> tuple[str, str]:
    """Object keys for the unified JSON and the flattened Markdown."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return (f"conversions/{doc_id}/{stem}.json", f"conversions/{doc_id}/{stem}.md")


def save_conversion_artifacts(doc_id: str, filename: str, document: dict) -> tuple[str, str, int]:
    """Persist unified JSON + Markdown to MinIO. Returns (json_key, md_key, bytes)."""
    from agent_test.rag import document_parser
    json_key, md_key = conversion_object_keys(doc_id, filename)
    payload = document_parser.dump_json(document)
    put_file_bytes(payload, json_key, "application/json")
    put_file_bytes((document.get("markdown") or "").encode("utf-8"), md_key, "text/markdown")
    logger.info("Conversion artifacts saved doc_id=%s json_key=%s bytes=%d", doc_id, json_key, len(payload))
    return json_key, md_key, len(payload)


def put_file_bytes(payload: bytes, object_name: str, content_type: str) -> str:
    from agent_test.rag import object_store
    from io import BytesIO
    store = object_store.ensure_bucket()
    store.put_object(settings.minio_bucket, object_name, BytesIO(payload),
                     length=len(payload), content_type=content_type)
    return object_name


def load_conversion_json(conversion: dict) -> dict:
    """Read a stored unified document JSON back from MinIO."""
    from agent_test.rag.object_store import client
    response = client().get_object(settings.minio_bucket, conversion["json_key"])
    try:
        return json.loads(response.read().decode("utf-8"))
    finally:
        response.close()
        response.release_conn()


def build_conversion_zip(conversions: list[dict]) -> bytes:
    """Bundle stored Markdown/JSON (and extracted images) into one ZIP."""
    from agent_test.rag.object_store import client
    store = client()
    buffer = io.BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for conversion in conversions:
            base = os.path.splitext(os.path.basename(conversion["filename"]))[0] or conversion["id"]
            folder = base
            suffix = 2
            while folder in used_names:
                folder = f"{base}-{suffix}"
                suffix += 1
            used_names.add(folder)
            for key, member in ((conversion.get("md_key"), f"{base}.md"),
                                (conversion.get("json_key"), f"{base}.json")):
                if not key:
                    continue
                try:
                    response = store.get_object(settings.minio_bucket, key)
                    try:
                        archive.writestr(f"{folder}/{member}", response.read())
                    finally:
                        response.close()
                        response.release_conn()
                except Exception as exc:
                    logger.warning("Conversion artifact missing key=%s error=%s", key, exc)
            # Images are embedded in the JSON as data URIs; export them as files too.
            try:
                document = load_conversion_json(conversion)
            except Exception:
                continue
            for image in document.get("images") or []:
                data_uri = image.get("data_uri") or ""
                if ";base64," not in data_uri:
                    continue
                import base64
                header, encoded = data_uri.split(";base64,", 1)
                ext = ".jpg" if "jpeg" in header or "jpg" in header else ".png"
                name = os.path.basename(image.get("name") or f"image{ext}")
                if not os.path.splitext(name)[1]:
                    name += ext
                try:
                    archive.writestr(f"{folder}/images/{name}", base64.b64decode(encoded))
                except Exception:
                    continue
    return buffer.getvalue()


def ingest_document(*, doc_id: str, filename: str, path: str | None, raw: bytes | None,
                    content_type: str, document: dict | None = None,
                    chunk_size: int | None = None, chunk_overlap: int | None = None,
                    task_id: str | None = None, embedding_model: str | None = None,
                    collection_name: str | None = None) -> dict:
    """Shared ingest body used by both the upload path and the conversion path.

    `document` short-circuits parsing so a file already converted by MinerU is
    not sent to the service twice.
    """
    from agent_test.rag import document_parser
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    split_text("validation", chunk_size, chunk_overlap)
    target_collection = collection_name or settings.collection
    target_embedding = embedding_model or settings.embedding_model

    def step(key: str, status: str, message: str):
        if task_id:
            update_ingest_task(task_id, key, status, message,
                               task_status="processing", document_id=doc_id)
        logger.info("Ingest step task_id=%s doc_id=%s step=%s status=%s message=%s",
                    task_id, doc_id, key, status, message)

    step("minio", "processing", "正在保存原文件")
    if raw is not None and path is not None:
        object_key = f"documents/{doc_id}/{os.path.basename(filename)}"
        put_file(path, object_key, content_type or "application/octet-stream")
    else:
        with connection() as db:
            row = db.execute("SELECT object_key FROM documents WHERE id=?", (doc_id,)).fetchone()
        object_key = (row["object_key"] if row else "") or f"documents/{doc_id}/{os.path.basename(filename)}"
    step("minio", "completed", "原文件已保存")

    step("split", "processing", "正在提取文本并切分")
    if document is None:
        document = parse_document(path or "", filename, raw)
    text = document_parser.document_text(document)
    chunks = split_text(text, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("No text was extracted from the uploaded document")
    converter = (document.get("source") or {}).get("converter", "")
    step("split", "completed",
         f"已通过 {converter} 解析 {document.get('metadata', {}).get('page_count', 0)} 页，"
         f"提取 {len(text)} 个字符，切分为 {len(chunks)} 段")

    step("embedding", "processing", f"正在使用 {target_embedding} 生成向量")
    from agent_test.rag.model_registry import registry
    vectors = registry.embed(chunks, embedding_model)
    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise ValueError("Embedding model returned inconsistent vector dimensions")
    dim = dims.pop()
    if settings.embedding_dimension and settings.embedding_dimension != dim:
        raise ValueError(f"Configured dimension {settings.embedding_dimension} differs from embedding output {dim}")
    step("embedding", "completed", f"已生成 {len(vectors)} 个向量，维度 {dim}")

    from agent_test.rag.storage import milvus_client
    client = milvus_client()
    step("collection", "processing", f"正在检查 collection：{target_collection}")
    index_created = ensure_collection(client, dim, target_collection)
    client.load_collection(target_collection)
    index_message = "缺失的向量索引已创建" if index_created else "向量索引已确认"
    step("collection", "completed", f"collection 已就绪并加载；{index_message}；维度 {dim}")

    step("persist", "processing", "正在写入文本片段和向量")
    count = insert_chunks(doc_id, filename, chunks, replace=False,
                          embedding_model=embedding_model, collection_name=target_collection,
                          vectors=vectors)
    step("persist", "completed", f"已写入 {count} 条记录")
    with connection() as db:
        db.execute("UPDATE documents SET object_key=?,status='ready',chunks=?,chunk_size=?,chunk_overlap=?,embedding_model=?,collection_name=? WHERE id=?",
                   (object_key, count, chunk_size, chunk_overlap, target_embedding,
                    target_collection, doc_id))
    logger.info("Document indexed doc_id=%s chunks=%d chunk_size=%d overlap=%d converter=%s",
                doc_id, count, chunk_size, chunk_overlap, converter)
    if task_id:
        update_ingest_task(task_id, "persist", "completed", f"已写入 {count} 条记录",
                           task_status="completed", document_id=doc_id)
    return {"id": doc_id, "name": filename, "status": "ready", "chunks": count,
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "duplicate": False,
            "converter": converter}


def ingest(path: str, filename: str, raw: bytes, content_type: str,
           chunk_size: int | None = None, chunk_overlap: int | None = None,
           *, task_id: str | None = None, embedding_model: str | None = None,
           collection_name: str | None = None) -> dict:
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    # Validate even when the file hash already exists.
    split_text("validation", chunk_size, chunk_overlap)
    doc_id = uuid.uuid4().hex
    def step(key: str, status: str, message: str):
        if task_id:
            update_ingest_task(task_id, key, status, message,
                               task_status="processing", document_id=doc_id)
        logger.info("Ingest step task_id=%s doc_id=%s step=%s status=%s message=%s",
                    task_id, doc_id, key, status, message)

    step("upload", "completed", f"已接收文件，大小 {len(raw)} 字节")
    digest = content_hash(raw)
    object_key = f"documents/{doc_id}/{os.path.basename(filename)}"
    with connection() as db:
        existing = db.execute("SELECT id,name,status,chunks,chunk_size,chunk_overlap,object_key FROM documents WHERE sha256=? ORDER BY created_at DESC LIMIT 1", (digest,)).fetchone()
        if existing and existing["status"] == "ready":
            logger.info("Duplicate document skipped name=%s existing_doc_id=%s", filename, existing["id"])
            if task_id:
                for key, label in [("minio", "原文件已存在于 MinIO"), ("split", "复用已处理文档"),
                                   ("embedding", "复用已有向量"), ("collection", "复用现有 collection"),
                                   ("persist", "跳过重复写入")]:
                    update_ingest_task(task_id, key, "skipped", label)
                update_ingest_task(task_id, "upload", "completed", "检测到重复文件，复用现有文档",
                                   task_status="completed", document_id=existing["id"])
            return dict(existing) | {"duplicate": True}
        if existing:
            doc_id = existing["id"]
            object_key = existing["object_key"] or object_key
            db.execute("UPDATE documents SET name=?,status='processing',chunks=0,error=NULL,chunk_size=?,chunk_overlap=?,embedding_model=?,collection_name=? WHERE id=?",
                       (filename, chunk_size, chunk_overlap, embedding_model or settings.embedding_model,
                        collection_name or settings.collection, doc_id))
        else:
            db.execute("INSERT INTO documents(id,name,object_key,sha256,status,created_at,chunk_size,chunk_overlap,embedding_model,collection_name) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (doc_id, filename, object_key, digest, "processing", datetime.now(timezone.utc).isoformat(),
                        chunk_size, chunk_overlap, embedding_model or settings.embedding_model,
                        collection_name or settings.collection))
    try:
        return ingest_document(doc_id=doc_id, filename=filename, path=path, raw=raw,
                               content_type=content_type, chunk_size=chunk_size,
                               chunk_overlap=chunk_overlap, task_id=task_id,
                               embedding_model=embedding_model, collection_name=collection_name)
    except Exception as exc:
        with connection() as db:
            db.execute("UPDATE documents SET status='error',error=? WHERE id=?", (str(exc)[:2000], doc_id))
        logger.exception("Document indexing failed doc_id=%s name=%s", doc_id, filename)
        if task_id:
            from agent_test.rag.usage_store import get_ingest_task
            task = get_ingest_task(task_id)
            failed_step = task["current_step"] if task else "persist"
            update_ingest_task(task_id, failed_step, "failed", str(exc)[:300],
                               task_status="failed", error=str(exc)[:2000], document_id=doc_id)
        raise


def reindex_document(doc_id: str, chunk_size: int, chunk_overlap: int) -> dict:
    with connection() as db:
        row = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not row:
        raise KeyError(doc_id)
    # MinIO原件下载到临时文件后重新解析，保证原始文件始终作为事实来源。
    from agent_test.rag.object_store import client
    os.makedirs(settings.upload_dir, exist_ok=True)
    temp_path = os.path.join(settings.upload_dir, f"reindex-{doc_id}-{os.path.basename(row['name'])}")
    try:
        client().fget_object(settings.minio_bucket, row["object_key"], temp_path)
        text = convert_document(temp_path, row["name"])
        chunks = split_text(text, chunk_size, chunk_overlap)
        if not chunks:
            raise ValueError("No text was extracted from the document")
        embedding_model = row["embedding_model"] or settings.embedding_model
        collection_name = row["collection_name"] or settings.collection
        count = insert_chunks(doc_id, row["name"], chunks, replace=True,
                              embedding_model=embedding_model, collection_name=collection_name)
        with connection() as db:
            db.execute("UPDATE documents SET status='ready',chunks=?,chunk_size=?,chunk_overlap=?,error=NULL WHERE id=?",
                       (count, chunk_size, chunk_overlap, doc_id))
        return {"id": doc_id, "name": row["name"], "status": "ready", "chunks": count,
                "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    except Exception as exc:
        with connection() as db:
            db.execute("UPDATE documents SET status='error',error=? WHERE id=?", (str(exc)[:2000], doc_id))
        raise
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def delete_document(doc_id: str) -> None:
    with connection() as db:
        row = db.execute("SELECT object_key,collection_name FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not row:
        raise KeyError(doc_id)
    delete_document_vectors(doc_id, row["collection_name"] or settings.collection)
    # Conversion-linked documents reuse the conversion's original file; keep
    # that object so the conversion record can still be downloaded/re-indexed.
    if not (row["object_key"] or "").startswith("conversions/"):
        remove_file(row["object_key"])
    with connection() as db:
        db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        db.execute("UPDATE conversions SET document_id=NULL WHERE document_id=?", (doc_id,))
