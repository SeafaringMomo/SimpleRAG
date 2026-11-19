"""Batch document conversion: parse to unified JSON/Markdown, then optionally index.

A conversion is a standalone artifact (unified JSON + Markdown + images in
MinIO) that can be downloaded as a ZIP or fed into the existing chunking /
embedding pipeline without parsing the original a second time.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import uuid

from agent_test.rag.config import settings
from agent_test.rag.document_parser import document_text
from agent_test.rag.document_service import (
    parse_document,
    put_file_bytes,
    save_conversion_artifacts,
    load_conversion_json,
)
from agent_test.rag.usage_store import (
    claim_conversion,
    connection,
    create_conversion,
    get_conversion,
    update_conversion,
)
from agent_test.rag.object_store import client as minio_client, ensure_bucket

logger = logging.getLogger(__name__)


def original_object_key(conversion_id: str, filename: str) -> str:
    return f"conversions/{conversion_id}/original/{os.path.basename(filename)}"


def run_conversion(conversion_id: str, path: str, filename: str, raw: bytes,
                   content_type: str) -> dict:
    """Parse one uploaded file and store the unified artifacts.

    Stages mirror the record's `stage` column so the UI can poll progress.
    """
    try:
        update_conversion(conversion_id, stage="saving_original", message="正在保存原文件")
        key = original_object_key(conversion_id, filename)
        ensure_bucket()
        put_file_bytes(raw, key, content_type or "application/octet-stream")

        update_conversion(conversion_id, stage="parsing", message="正在解析文档")

        def report(task: dict) -> None:
            # MinerU reports queue position and processing state while we poll.
            queued = task.get("queued_ahead")
            status = task.get("status") or ""
            if status == "processing":
                message = "MinerU 正在解析"
            elif isinstance(queued, int) and queued > 0:
                message = f"MinerU 排队中，前方 {queued} 个任务"
            else:
                message = "已提交 MinerU，等待开始"
            update_conversion(conversion_id, stage="parsing", message=message)

        document = parse_document(path, filename, raw, progress=report)
        converter = (document.get("source") or {}).get("converter", "")
        if not document_text(document):
            raise ValueError("No text was extracted from the document")

        update_conversion(conversion_id, stage="writing", message="正在生成 JSON 与 Markdown")
        json_key, md_key, size = save_conversion_artifacts(conversion_id, filename, document)
        meta = document.get("metadata") or {}
        update_conversion(
            conversion_id,
            status="converted",
            stage="converted",
            message=f"已转换：{meta.get('page_count', 0)} 页 / {meta.get('block_count', 0)} 块 / "
                    f"{meta.get('char_count', 0)} 字符",
            converter=converter,
            object_key=key,
            json_key=json_key,
            md_key=md_key,
            sha256=(document.get("source") or {}).get("sha256", ""),
            page_count=int(meta.get("page_count", 0)),
            block_count=int(meta.get("block_count", 0)),
            char_count=int(meta.get("char_count", 0)),
            image_count=int(meta.get("image_count", 0)),
            error="",
        )
        logger.info("Conversion completed id=%s filename=%s converter=%s pages=%d bytes=%d",
                    conversion_id, filename, converter, meta.get("page_count", 0), size)
        return get_conversion(conversion_id) or {}
    except Exception as exc:
        logger.exception("Conversion failed id=%s filename=%s", conversion_id, filename)
        update_conversion(conversion_id, status="failed", stage="failed",
                          message="转换失败，可重试", error=str(exc)[:2000])
        raise


def retry_conversion(conversion_id: str, *, already_claimed: bool = False) -> dict:
    """Re-run a failed conversion from its stored original file.

    The uploaded original lives in MinIO, so a retry never needs the browser to
    re-upload. Artifacts are overwritten in place (same object keys).
    """
    conversion = get_conversion(conversion_id)
    if not conversion:
        raise KeyError(conversion_id)
    if not already_claimed and not claim_conversion(conversion_id, stage="received",
                                                    message="重新解析中"):
        raise ValueError("Conversion is already running")
    key = conversion.get("object_key") or original_object_key(conversion_id, conversion["filename"])
    from agent_test.rag.object_store import client as minio_client
    os.makedirs(settings.upload_dir, exist_ok=True)
    suffix = os.path.splitext(conversion["filename"])[1]
    temp_path = os.path.join(settings.upload_dir, f"retry-{conversion_id}{suffix}")
    response = minio_client().get_object(settings.minio_bucket, key)
    try:
        payload = response.read()
    finally:
        response.close()
        response.release_conn()
    try:
        with open(temp_path, "wb") as handle:
            handle.write(payload)
        return run_conversion(conversion_id, temp_path, conversion["filename"], payload,
                              mimetypes.guess_type(conversion["filename"])[0]
                              or "application/octet-stream")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def conversion_preview(conversion: dict, limit: int = 4000) -> dict:
    """Markdown + block outline for the in-browser preview panel."""
    document = load_conversion_json(conversion)
    markdown = document.get("markdown") or ""
    outline = []
    for page in document.get("pages") or []:
        for block in page.get("blocks") or []:
            if block.get("type") == "title" and block.get("text"):
                outline.append({"level": int(block.get("level") or 1),
                                "text": block["text"],
                                "page_idx": page.get("page_idx", 0)})
    return {"markdown": markdown[:limit], "truncated": len(markdown) > limit,
            "total_chars": len(markdown), "outline": outline[:100],
            "page_count": (document.get("metadata") or {}).get("page_count", 0)}


def index_conversion(conversion_id: str, chunk_size: int, chunk_overlap: int,
                     *, task_id: str, embedding_model: str | None = None,
                     collection_name: str | None = None) -> dict:
    """Chunk + embed an already-converted document.

    The original file is not parsed again: the stored unified JSON is reused and
    the document row points at the stored original so reindex still works.
    """
    from agent_test.rag.document_service import ingest_document
    from agent_test.rag.usage_store import update_ingest_task

    conversion = get_conversion(conversion_id)
    if not conversion:
        raise KeyError(conversion_id)
    if conversion["status"] not in {"converted", "index_failed", "indexed"}:
        raise ValueError(f"Conversion {conversion_id} is {conversion['status']}, not converted")
    if not conversion["json_key"]:
        raise ValueError(f"Conversion {conversion_id} has no stored JSON")

    document = load_conversion_json(conversion)
    doc_id = uuid.uuid4().hex
    filename = conversion["filename"]
    object_key = conversion["object_key"] or original_object_key(conversion_id, filename)
    # Re-indexing the same conversion replaces the previous document row + vectors.
    # The original object is shared with the conversion, so it must not be removed.
    if conversion.get("document_id"):
        from agent_test.rag.embedding_service import delete_document_vectors
        old_id = conversion["document_id"]
        with connection() as db:
            old = db.execute("SELECT object_key,collection_name FROM documents WHERE id=?",
                             (old_id,)).fetchone()
        if old:
            try:
                delete_document_vectors(old_id, old["collection_name"] or settings.collection)
            except Exception as exc:
                logger.warning("Old conversion vectors not removed doc_id=%s error=%s", old_id, exc)
            old_key = old["object_key"] or ""
            if old_key and old_key != object_key:
                from agent_test.rag.object_store import remove_file
                try:
                    remove_file(old_key)
                except Exception as exc:
                    logger.warning("Old document object not removed key=%s error=%s", old_key, exc)
            with connection() as db:
                db.execute("DELETE FROM documents WHERE id=?", (old_id,))
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with connection() as db:
        db.execute(
            "INSERT INTO documents(id,name,object_key,sha256,status,created_at,chunk_size,chunk_overlap,embedding_model,collection_name) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (doc_id, filename, object_key, conversion["sha256"] or "", "processing", now,
             chunk_size, chunk_overlap, embedding_model or settings.embedding_model,
             collection_name or settings.collection))
    update_conversion(conversion_id, status="indexing", stage="indexing",
                      message="正在切分并写入向量库", document_id=doc_id)
    update_ingest_task(task_id, "upload", "completed",
                       f"复用已转换文档，大小 {conversion.get('char_count', 0)} 字符")
    update_ingest_task(task_id, "minio", "completed", "原文件已位于 MinIO")
    try:
        result = ingest_document(doc_id=doc_id, filename=filename, path=None, raw=None,
                                 content_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
                                 document=document, chunk_size=chunk_size,
                                 chunk_overlap=chunk_overlap, task_id=task_id,
                                 embedding_model=embedding_model,
                                 collection_name=collection_name)
    except Exception as exc:
        with connection() as db:
            db.execute("UPDATE documents SET status='error',error=? WHERE id=?",
                       (str(exc)[:2000], doc_id))
        from agent_test.rag.usage_store import get_ingest_task
        task = get_ingest_task(task_id)
        failed_step = task["current_step"] if task else "persist"
        update_ingest_task(task_id, failed_step, "failed", str(exc)[:300],
                           task_status="failed", error=str(exc)[:2000], document_id=doc_id)
        update_conversion(conversion_id, status="converted", stage="index_failed",
                          message="切分入库失败，可重试", error=str(exc)[:2000],
                          document_id=doc_id)
        raise
    update_conversion(conversion_id, status="indexed", stage="indexed",
                      message=f"已入库 {result['chunks']} 个片段", document_id=doc_id)
    return result


def delete_conversion(conversion_id: str) -> None:
    """Remove the conversion record and its stored artifacts (not indexed vectors)."""
    conversion = get_conversion(conversion_id)
    if not conversion:
        raise KeyError(conversion_id)
    store = minio_client()
    for key in (conversion.get("object_key"), conversion.get("json_key"),
                conversion.get("md_key")):
        if not key:
            continue
        try:
            store.remove_object(settings.minio_bucket, key)
        except Exception as exc:
            logger.warning("Conversion artifact removal failed key=%s error=%s", key, exc)
    from agent_test.rag.usage_store import delete_conversion as _delete
    _delete(conversion_id)
