from __future__ import annotations

import os
import tempfile
import time
import logging
import threading
import uuid
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


from agent_test.rag.config import settings
from agent_test.rag.model_registry import registry
from agent_test.rag.usage_store import (claim_conversion, create_conversion, create_ingest_task,
                         get_conversion, get_ingest_task, get_setting, init_db,
                         list_conversions, list_documents, recent_ingest_tasks, set_settings,
                         update_ingest_task, usage_summary)
from agent_test.rag.document_service import ingest
from agent_test.rag.chat_service import answer
from agent_test.rag.logging_config import configure_logging
from agent_test.rag.document_service import (build_conversion_zip, delete_document,
                                             reindex_document)
from agent_test.rag.conversion_service import (conversion_preview, delete_conversion,
                                               index_conversion, retry_conversion,
                                               run_conversion)

configure_logging()
logger = logging.getLogger("rag-workbench")
health_cache: dict = {"time": 0.0, "value": None}
health_lock = threading.Lock()
# Conversions are parsed on a bounded pool instead of Starlette BackgroundTasks:
# a batch of 20 PDFs would otherwise all hit MinerU at once and sit in its queue,
# and in-flight work would be lost between requests on reload.
conversion_pool = ThreadPoolExecutor(max_workers=max(1, settings.conversion_workers),
                                     thread_name_prefix="convert")


def submit_conversion(conversion_id: str, path: str, filename: str, payload: bytes,
                      content_type: str) -> None:
    def job():
        try:
            run_conversion(conversion_id, path, filename, payload, content_type)
        except Exception:
            logger.exception("Conversion task failed conversion_id=%s", conversion_id)
        finally:
            if os.path.exists(path):
                os.unlink(path)
    conversion_pool.submit(job)




@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    os.makedirs(settings.upload_dir, exist_ok=True)
    recover_stale_conversions()
    yield
    conversion_pool.shutdown(wait=False, cancel_futures=True)


def recover_stale_conversions() -> None:
    """Mark conversions interrupted by a restart as failed so they can be retried.

    Conversion jobs live in an in-process pool; whatever was in flight when the
    process died will never finish, and a record stuck at `processing` blocks
    both retry and the stale-job guard in `index_conversion`.
    """
    from agent_test.rag.usage_store import list_conversions, update_conversion as _update
    stale = [c for c in list_conversions(limit=500) if c["status"] in {"processing", "indexing"}]
    for conversion in stale:
        if conversion["status"] == "indexing" and conversion.get("json_key"):
            # Artifacts already exist; only the vector write was cut off, so a
            # re-index is enough and re-parsing would be wasted work.
            _update(conversion["id"], status="converted", stage="index_failed",
                    message="服务重启，入库中断，可重新入库",
                    error="服务重启，入库未完成")
        else:
            _update(conversion["id"], status="failed", stage="failed",
                    message="服务重启，转换中断，可重试", error="服务重启，任务未完成")
    if stale:
        logger.warning("Recovered %d interrupted conversions", len(stale))


APP_VERSION = "v0.1.0"

app = FastAPI(title="RAG Model Workbench", version=APP_VERSION, lifespan=lifespan)
app.mount("/web", StaticFiles(directory="./web"), name="web")

@app.middleware("http")
async def request_log(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    started = time.monotonic()
    try:
        response = await call_next(request)
        logger.info("request_id=%s method=%s path=%s status=%d latency_ms=%d",
                    request_id, request.method, request.url.path, response.status_code,
                    int((time.monotonic() - started) * 1000))
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.exception("request_id=%s method=%s path=%s failed",
                         request_id, request.method, request.url.path)
        raise


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=30000)
    use_rag: bool = False
    top_k: int = Field(default=5, ge=1, le=30)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    chat_model: str | None = None
    embedding_model: str | None = None
    collection_name: str | None = None


class EmbedRequest(BaseModel):
    text: str = Field(min_length=1, max_length=30000)


class ReindexRequest(BaseModel):
    chunk_size: int = Field(ge=100, le=20000)
    chunk_overlap: int = Field(ge=0, le=10000)


class ConversionIndexRequest(BaseModel):
    chunk_size: int = Field(ge=100, le=20000)
    chunk_overlap: int = Field(ge=0, le=10000)


class ConversionDownloadRequest(BaseModel):
    ids: list[str] = Field(min_length=1)


class ModelSelection(BaseModel):
    chat_model: str
    embedding_model: str


class StorageSettings(BaseModel):
    milvus_host: str = Field(min_length=1, max_length=255)
    milvus_port: int = Field(ge=1, le=65535)
    minio_host: str = Field(min_length=1, max_length=255)
    minio_port: int = Field(ge=1, le=65535)


@app.get("/")
async def home():
    return FileResponse("./web/index.html")


@app.get("/api/config")
async def api_config():
    models = [{"name": m.name, "category": m.category, "api_style": m.api_style} for m in settings.models]
    from agent_test.rag import mineru_client
    return {"chat_model": get_setting("chat_model", settings.chat_model),
            "embedding_model": get_setting("embedding_model", settings.embedding_model),
            "collection_name": settings.collection, "models": models,
            "upload_max_mb": settings.max_upload_mb,
            "mineru_configured": bool(settings.mineru_url),
            "mineru_extensions": sorted(mineru_client.mineru_supported_extensions()),
            "mineru_async": settings.mineru_async,
            "conversion_workers": settings.conversion_workers}


@app.put("/api/settings/models")
async def save_model_selection(selection: ModelSelection):
    validate_model_selection(selection.chat_model, selection.embedding_model)
    set_settings({"chat_model": selection.chat_model, "embedding_model": selection.embedding_model})
    logger.info("Model selection updated chat_model=%s embedding_model=%s", selection.chat_model, selection.embedding_model)
    return {"chat_model": selection.chat_model, "embedding_model": selection.embedding_model}


@app.get("/api/storage")
async def storage_settings():
    """Current MinIO/Milvus endpoints (runtime overrides + .env defaults)."""
    from agent_test.rag import storage
    return {**storage.current(), "version": APP_VERSION}


@app.put("/api/storage")
async def save_storage_settings(req: StorageSettings):
    """Persist new endpoints to app_settings; .env itself is never modified."""
    from agent_test.rag import storage
    return storage.save(req.milvus_host, req.milvus_port, req.minio_host, req.minio_port)


@app.post("/api/storage/probe")
async def probe_storage():
    """Connect to both services and create the bucket/database when missing."""
    from agent_test.rag import storage
    try:
        return await asyncio.wait_for(asyncio.to_thread(storage.probe), timeout=25)
    except asyncio.TimeoutError:
        raise HTTPException(504, "Storage probe timed out; check the host/port")


@app.post("/api/storage/reset")
async def reset_storage_settings():
    """Drop runtime overrides so the .env endpoints are used again."""
    from agent_test.rag import storage
    from agent_test.rag.usage_store import delete_settings
    delete_settings([storage.MILVUS_HOST_KEY, storage.MILVUS_PORT_KEY,
                     storage.MINIO_HOST_KEY, storage.MINIO_PORT_KEY])
    return storage.current()


@app.get("/api/health")
async def health():
    now = time.monotonic()
    if health_cache["value"] is not None and now - health_cache["time"] < settings.health_cache_seconds:
        return health_cache["value"]
    with health_lock:
        now = time.monotonic()
        if health_cache["value"] is not None and now - health_cache["time"] < settings.health_cache_seconds:
            return health_cache["value"]
        value = {"models": registry.health(), "services": await service_health()}
        health_cache.update(time=now, value=value)
        return value


async def service_health():
    services = []
    try:
        from agent_test.rag import storage
        result = await asyncio.wait_for(asyncio.to_thread(storage.probe), timeout=25)
        services.extend([
            {"name": "Milvus", "status": "connected" if result["milvus"]["status"] == "connected" else "error",
             "error": result["milvus"].get("error")},
            {"name": "MinIO", "status": "connected" if result["minio"]["status"] == "connected" else "error",
             "error": result["minio"].get("error")},
        ])
    except Exception as exc:
        services.extend([{"name": "Milvus", "status": "error", "error": str(exc)[:300]},
                         {"name": "MinIO", "status": "error", "error": str(exc)[:300]}])
    try:
        from agent_test.rag import mineru_client
        services.append(mineru_client.health())
    except Exception as exc:
        services.append({"name": "MinerU", "status": "error", "error": str(exc)[:300]})
    return services


@app.get("/api/usage")
async def usage():
    return usage_summary()


@app.get("/api/logs/recent-errors")
async def recent_errors(limit: int = 50):
    limit = max(1, min(limit, 200))
    try:
        with open(settings.log_path, "r", encoding="utf-8") as log_file:
            lines = log_file.readlines()
    except FileNotFoundError:
        return {"path": settings.log_path, "errors": []}
    errors = []
    for line in lines:
        if " ERROR " not in line and " CRITICAL " not in line:
            continue
        match = re.match(r"^(\S+\s+\S+)\s+(ERROR|CRITICAL)\s+([^ ]+)(?:\s+\[[^]]+\])?\s+(.*)$", line.rstrip())
        if match:
            # Exception messages can contain prompts, URLs, or provider details.
            summary = "Exception recorded; see local log file for details." if "Traceback" in match.group(4) else re.sub(
                r"(api[_-]?key|token|authorization)\s*[:=]\s*\S+", r"\1=[redacted]", match.group(4), flags=re.I
            )
            errors.append({"time": match.group(1), "level": match.group(2),
                           "logger": match.group(3), "summary": summary[:180]})
        else:
            errors.append({"summary": "Error recorded; see local log file for details."})
    return {"path": settings.log_path, "errors": errors[-limit:]}


@app.get("/api/documents")
async def documents():
    return list_documents()


@app.get("/api/ingest-tasks")
async def ingest_tasks():
    return recent_ingest_tasks()


@app.get("/api/ingest-tasks/{task_id}")
async def ingest_task(task_id: str):
    task = get_ingest_task(task_id)
    if not task:
        raise HTTPException(404, "Ingest task not found")
    return task


def run_ingest_task(task_id: str, path: str, filename: str, payload: bytes,
                    content_type: str, chunk_size: int, chunk_overlap: int,
                    embedding_model: str, collection_name: str):
    try:
        result = ingest(path, filename, payload, content_type, chunk_size, chunk_overlap,
                        task_id=task_id, embedding_model=embedding_model,
                        collection_name=collection_name)
        logger.info("Ingest task completed task_id=%s document_id=%s", task_id, result.get("id"))
    except Exception:
        logger.exception("Ingest task failed task_id=%s", task_id)
    finally:
        if os.path.exists(path):
            os.unlink(path)


# 根据模型类别对模型进行分类
def validate_model_selection(chat_model: str, embedding_model: str) -> None:
    chat_models = {m.name for m in settings.models if m.api_style.lower() in {"chat"}}
    embedding_models = {m.name for m in settings.models if m.api_style.lower() in {"embedding"}}
    if chat_model not in chat_models:
        raise HTTPException(422, "Selected chat model is not configured as a chat model")
    if embedding_model not in embedding_models:
        raise HTTPException(422, "Selected embedding model is not configured as an embedding model")


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: str):
    try:
        delete_document(doc_id)
        logger.info("Document deleted doc_id=%s", doc_id)
        return {"success": True, "id": doc_id}
    except KeyError as exc:
        raise HTTPException(404, "Document not found") from exc
    except Exception as exc:
        logger.exception("Document deletion failed doc_id=%s", doc_id)
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/documents/{doc_id}/reindex")
async def reindex(doc_id: str, req: ReindexRequest):
    try:
        if req.chunk_overlap >= req.chunk_size:
            raise HTTPException(422, "chunk_overlap must be smaller than chunk_size")
        result = reindex_document(doc_id, req.chunk_size, req.chunk_overlap)
        logger.info("Document reindexed doc_id=%s chunks=%d chunk_size=%d overlap=%d",
                    doc_id, result["chunks"], req.chunk_size, req.chunk_overlap)
        return result
    except KeyError as exc:
        raise HTTPException(404, "Document not found") from exc
    except Exception as exc:
        logger.exception("Document reindex failed doc_id=%s", doc_id)
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(settings.chunk_size, ge=100, le=20000),
    chunk_overlap: int = Form(settings.chunk_overlap, ge=0, le=10000),
    background_tasks: BackgroundTasks = None,
):
    filename = os.path.basename(file.filename or "upload")
    allowed = {".docx", ".txt", ".md", ".markdown", ".xlsx", ".xls", ".csv", ".pptx", ".html", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(415, f"Unsupported file type: {ext}")
    if chunk_overlap >= chunk_size:
        raise HTTPException(422, "chunk_overlap must be smaller than chunk_size")
    payload = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Maximum upload size is {settings.max_upload_mb} MB")
    suffix = ext
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.upload_dir) as temp:
            temp.write(payload)
            temp_path = temp.name
        task_id = uuid.uuid4().hex
        create_ingest_task(task_id, filename)
        update_ingest_task(task_id, "upload", "completed", f"已接收文件，大小 {len(payload)} 字节")
        chat_model = get_setting("chat_model", settings.chat_model)
        embedding_model = get_setting("embedding_model", settings.embedding_model)
        # Starlette BackgroundTasks runs after the response and retains the temp file until processing ends.
        background_tasks.add_task(run_ingest_task, task_id, temp_path, filename, payload,
                                  file.content_type or "application/octet-stream", chunk_size,
                                  chunk_overlap, embedding_model, settings.collection)
        temp_path = None
        return {"task_id": task_id, "filename": filename, "status": "processing",
                "chat_model": chat_model, "embedding_model": embedding_model,
                "collection_name": settings.collection}
    except Exception as exc:
        logger.exception("Document ingest failed")
        raise HTTPException(502, str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


CONVERSION_ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
                      ".docx", ".doc", ".txt", ".md", ".markdown", ".xlsx", ".xls", ".csv",
                      ".json", ".log", ".pptx", ".html", ".htm"}


@app.get("/api/conversions")
async def conversions(batch_id: str | None = None, limit: int = 50):
    from agent_test.rag import mineru_client
    return {"mineru_configured": bool(settings.mineru_url),
            "mineru_async": settings.mineru_async,
            "mineru_extensions": sorted(mineru_client.mineru_supported_extensions()),
            "items": list_conversions(limit=max(1, min(limit, 200)), batch_id=batch_id)}


@app.post("/api/conversions")
async def upload_conversions(
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
):
    """Batch-convert files to the unified JSON/Markdown artifacts."""
    if not files:
        raise HTTPException(422, "No files uploaded")
    batch_id = uuid.uuid4().hex
    limit_bytes = settings.max_upload_mb * 1024 * 1024
    accepted, rejected = [], []
    for file in files:
        filename = os.path.basename(file.filename or "upload")
        ext = os.path.splitext(filename)[1].lower()
        if ext not in CONVERSION_ALLOWED:
            rejected.append({"filename": filename, "error": f"Unsupported file type: {ext}"})
            continue
        payload = await file.read(limit_bytes + 1)
        if len(payload) > limit_bytes:
            rejected.append({"filename": filename,
                             "error": f"Maximum upload size is {settings.max_upload_mb} MB"})
            continue
        conversion_id = uuid.uuid4().hex
        temp_path = None
        try:
            os.makedirs(settings.upload_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext,
                                             dir=settings.upload_dir) as temp:
                temp.write(payload)
                temp_path = temp.name
            create_conversion(conversion_id, batch_id, filename)
            submit_conversion(conversion_id, temp_path, filename, payload,
                              file.content_type or "application/octet-stream")
            temp_path = None
            accepted.append({"id": conversion_id, "filename": filename})
        except Exception as exc:
            logger.exception("Conversion upload failed filename=%s", filename)
            rejected.append({"filename": filename, "error": str(exc)[:300]})
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    return {"batch_id": batch_id, "items": accepted, "rejected": rejected}


@app.get("/api/conversions/{conversion_id}")
async def conversion_detail(conversion_id: str):
    conversion = get_conversion(conversion_id)
    if not conversion:
        raise HTTPException(404, "Conversion not found")
    return conversion


@app.get("/api/conversions/{conversion_id}/preview")
async def conversion_preview_route(conversion_id: str, limit: int = 4000):
    """Markdown head + heading outline for the in-browser preview."""
    conversion = get_conversion(conversion_id)
    if not conversion:
        raise HTTPException(404, "Conversion not found")
    if not conversion.get("json_key"):
        raise HTTPException(409, "Conversion has no stored artifacts yet")
    limit = max(200, min(limit, 50000))
    try:
        return conversion_preview(conversion, limit=limit)
    except Exception as exc:
        logger.exception("Conversion preview failed conversion_id=%s", conversion_id)
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/conversions/{conversion_id}/retry")
async def conversion_retry(conversion_id: str):
    """Re-parse a failed conversion from the original stored in MinIO."""
    conversion = get_conversion(conversion_id)
    if not conversion:
        raise HTTPException(404, "Conversion not found")
    if conversion["status"] in {"processing", "indexing"}:
        raise HTTPException(409, f"Conversion is {conversion['status']}, wait for it to finish")
    try:
        # Claim atomically so a double-clicked retry cannot queue two parses.
        if not claim_conversion(conversion_id, stage="received", message="排队重新解析"):
            raise HTTPException(409, f"Conversion is {conversion['status']}, wait for it to finish")
        conversion_pool.submit(guarded_retry, conversion_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Conversion retry failed conversion_id=%s", conversion_id)
        raise HTTPException(502, str(exc)) from exc
    return {"id": conversion_id, "status": "processing"}


def guarded_retry(conversion_id: str) -> None:
    try:
        retry_conversion(conversion_id, already_claimed=True)
    except Exception:
        logger.exception("Conversion retry task failed conversion_id=%s", conversion_id)


@app.post("/api/conversions/{conversion_id}/index")
async def conversion_index(conversion_id: str, req: ConversionIndexRequest,
                           background_tasks: BackgroundTasks = None):
    """Chunk + embed an already-converted document (no re-parse)."""
    conversion = get_conversion(conversion_id)
    if not conversion:
        raise HTTPException(404, "Conversion not found")
    if conversion["status"] not in {"converted", "index_failed", "indexed"}:
        raise HTTPException(409, f"Conversion is {conversion['status']}, wait for conversion to finish")
    if req.chunk_overlap >= req.chunk_size:
        raise HTTPException(422, "chunk_overlap must be smaller than chunk_size")
    task_id = uuid.uuid4().hex
    create_ingest_task(task_id, conversion["filename"])
    embedding_model = get_setting("embedding_model", settings.embedding_model)
    def job():
        try:
            index_conversion(conversion_id, req.chunk_size, req.chunk_overlap,
                             task_id=task_id, embedding_model=embedding_model)
        except Exception:
            logger.exception("Conversion indexing failed conversion_id=%s", conversion_id)

    # Indexing embeds locally/remotely and writes to Milvus; it is not MinerU
    # bound, so give it its own thread rather than queueing behind conversions.
    threading.Thread(target=job, name=f"index-{task_id[:8]}", daemon=True).start()
    return {"task_id": task_id, "conversion_id": conversion_id, "status": "processing",
            "embedding_model": embedding_model, "collection_name": settings.collection}


def zip_headers(filename: str) -> dict:
    """Attachment headers that survive non-ASCII names (RFC 5987)."""
    from urllib.parse import quote
    fallback = re.sub(r"[^\x20-\x7e]", "_", filename) or "converted.zip"
    return {"Content-Disposition":
            f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"}


@app.get("/api/conversions/{conversion_id}/download")
async def conversion_download(conversion_id: str):
    conversion = get_conversion(conversion_id)
    if not conversion:
        raise HTTPException(404, "Conversion not found")
    if not conversion.get("json_key"):
        raise HTTPException(409, "Conversion has no stored artifacts yet")
    try:
        payload = build_conversion_zip([conversion])
    except Exception as exc:
        logger.exception("Conversion zip failed conversion_id=%s", conversion_id)
        raise HTTPException(502, str(exc)) from exc
    stem = os.path.splitext(conversion["filename"])[0] or "converted"
    return Response(payload, media_type="application/zip",
                    headers=zip_headers(f"{stem}.zip"))


@app.post("/api/conversions/download")
async def conversion_download_batch(req: ConversionDownloadRequest):
    selected = [get_conversion(cid) for cid in req.ids]
    selected = [item for item in selected if item and item.get("json_key")]
    if not selected:
        raise HTTPException(404, "No converted artifacts found for the selected conversions")
    try:
        payload = build_conversion_zip(selected)
    except Exception as exc:
        logger.exception("Batch conversion zip failed")
        raise HTTPException(502, str(exc)) from exc
    return Response(payload, media_type="application/zip",
                    headers=zip_headers("conversions.zip"))


@app.delete("/api/conversions/{conversion_id}")
async def remove_conversion(conversion_id: str):
    try:
        delete_conversion(conversion_id)
        return {"success": True, "id": conversion_id}
    except KeyError as exc:
        raise HTTPException(404, "Conversion not found") from exc
    except Exception as exc:
        logger.exception("Conversion deletion failed conversion_id=%s", conversion_id)
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        if req.use_rag:
            chat_model = req.chat_model or get_setting("chat_model", settings.chat_model)
            embedding_model = req.embedding_model or get_setting("embedding_model", settings.embedding_model)
            validate_model_selection(chat_model, embedding_model)
            result = answer(req.message, req.top_k, req.temperature, req.max_tokens,
                            chat_model, embedding_model, req.collection_name or settings.collection)
            return {"model": chat_model, "embedding_model": embedding_model, **result}
        options = {}
        if req.temperature is not None:
            options["temperature"] = req.temperature
        if req.max_tokens is not None:
            options["max_tokens"] = req.max_tokens
        chat_model = req.chat_model or get_setting("chat_model", settings.chat_model)
        if chat_model not in {m.name for m in settings.models if m.category.lower() in {"deepseek", "chat", "llm"}}:
            raise HTTPException(422, "Selected chat model is not configured as a chat model")
        result = registry.chat(chat_model, [{"role": "user", "content": req.message}], task="chat", **options)
        return {"model": chat_model, "answer": result.choices[0].message.content,
                "usage": result.usage.model_dump() if result.usage else None, "sources": []}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/embed")
async def embed(req: EmbedRequest):
    try:
        vectors = registry.embed([req.text])
        return {"model": settings.embedding_model, "dimensions": len(vectors[0]), "preview": vectors[0][:8]}
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/health/check")
async def check_health():
    return await health()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), reload=False)
