"""Client for the MinerU FastAPI parsing service (`protocol_version` 2).

MinerU returns, for each input file, `md_content`, `content_list` (a JSON
*string*) and base64 images. This module turns that into the unified document
format defined in `document_parser` and offers both the synchronous
`POST /file_parse` and the asynchronous `POST /tasks` + poll protocol.

MinerU accepts PDFs, images, DOCX, PPTX and XLSX (see
`MINERU_NATIVE_EXTENSIONS`); anything else never reaches it.
"""
from __future__ import annotations

import json
import logging
import os
import time

import httpx

from agent_test.rag.config import settings
from agent_test.rag.document_parser import build_mineru_document

logger = logging.getLogger(__name__)

# Formats the service parses itself (its `pdf_suffixes + office_suffixes + image_suffixes`).
MINERU_NATIVE_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
    ".docx", ".pptx", ".xlsx",
}


class MinerUError(RuntimeError):
    """Raised when the MinerU service is unreachable or returns an error."""


class MinerUTimeout(MinerUError):
    """Raised when a submitted task does not finish inside `MINERU_TIMEOUT`."""


def mineru_available() -> bool:
    return bool(settings.mineru_url)


def configured_extensions() -> set[str]:
    return {e.strip().lower() for e in settings.mineru_extensions.split(",") if e.strip()}


def mineru_supported_extensions() -> set[str]:
    """Extensions this deployment will actually hand to MinerU."""
    return configured_extensions() & MINERU_NATIVE_EXTENSIONS


def should_use_mineru(filename: str) -> bool:
    """MinerU is used when configured and the extension routes to it.

    MinerU rejects formats outside its native set, so only the intersection of
    the configured list and those formats is eligible.
    """
    if not mineru_available():
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in mineru_supported_extensions()


def _form_bool(value: bool) -> str:
    return "true" if value else "false"


def _params() -> dict[str, str]:
    """Form fields shared by /file_parse and /tasks."""
    return {
        "backend": settings.mineru_backend,
        "effort": settings.mineru_effort,
        "parse_method": settings.mineru_parse_method,
        "lang_list": settings.mineru_lang,
        "formula_enable": _form_bool(settings.mineru_formula_enable),
        "table_enable": _form_bool(settings.mineru_table_enable),
        "image_analysis": _form_bool(settings.mineru_image_analysis),
        "return_md": "true",
        "return_content_list": "true",
        "return_images": "true",
        "return_middle_json": "false",
        "return_model_output": "false",
    }


def _files(path: str, filename: str) -> dict:
    with open(path, "rb") as handle:
        payload = handle.read()
    return {"files": (os.path.basename(filename), payload, "application/octet-stream")}


def _parse_entry(entry, filename: str) -> tuple[str, list[dict], dict[str, str]]:
    """Extract (markdown, content_list, images) from one MinerU result entry."""
    if not isinstance(entry, dict):
        return "", [], {}
    markdown = entry.get("md_content") or ""
    raw_list = entry.get("content_list")
    if isinstance(raw_list, str):
        try:
            raw_list = json.loads(raw_list) if raw_list.strip() else []
        except json.JSONDecodeError:
            logger.warning("MinerU returned unparsable content_list filename=%s", filename)
            raw_list = []
    if not isinstance(raw_list, list):
        raw_list = []
    images = entry.get("images") if isinstance(entry.get("images"), dict) else {}
    return markdown, [item for item in raw_list if isinstance(item, dict)], images


def _result_entry(body: dict, filename: str):
    """Pick this file's entry out of a `results` mapping.

    The service keys results by the uploaded file stem, so two files with the
    same stem in one batch collide; single-file uploads are the norm here.
    """
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, dict) or not results:
        raise MinerUError("MinerU response contained no results")
    stem = os.path.splitext(os.path.basename(filename))[0]
    entry = results.get(stem)
    if entry is None:
        entry = results.get(os.path.basename(filename))
    if entry is None:
        entry = next(iter(results.values()))
    return entry


def _document_from_body(body: dict, filename: str, raw: bytes | None, started: float) -> dict:
    markdown, content_list, images = _parse_entry(_result_entry(body, filename), filename)
    if not markdown and not content_list:
        raise MinerUError("MinerU returned neither markdown nor content_list")
    document = build_mineru_document(
        filename, raw, content_list, markdown,
        backend=body.get("backend") or settings.mineru_backend,
        version=str(body.get("version") or ""),
        images=images,
    )
    logger.info("MinerU parsed filename=%s pages=%d blocks=%d chars=%d images=%d backend=%s latency_ms=%d",
                filename, document["metadata"]["page_count"], document["metadata"]["block_count"],
                document["metadata"]["char_count"], document["metadata"]["image_count"],
                document["metadata"].get("backend", ""),
                int((time.monotonic() - started) * 1000))
    return document


def _check_response_status(response: httpx.Response) -> None:
    if response.status_code not in (200, 201, 202):
        raise MinerUError(f"MinerU returned HTTP {response.status_code}: {response.text[:500]}")
    if "application/json" not in response.headers.get("content-type", ""):
        raise MinerUError("MinerU returned a non-JSON response (ZIP output is not enabled)")


def parse_with_mineru(path: str, filename: str, raw: bytes | None = None,
                      *, timeout: int | None = None,
                      progress=None) -> dict:
    """Send one file to MinerU and return a unified document.

    With `MINERU_ASYNC` on this submits `POST /tasks` and polls the status
    endpoint, which keeps one connection from being held open for minutes and
    lets `progress(queued, status)` report queue position to the UI. Otherwise
    it uses the blocking `POST /file_parse`.

    Raises MinerUError for transport failures and service errors so callers can
    decide whether to fall back to a local converter.
    """
    if not mineru_available():
        raise MinerUError("MINERU_URL is not configured")
    if settings.mineru_async:
        return _parse_async(path, filename, raw, timeout=timeout, progress=progress)
    return _parse_sync(path, filename, raw, timeout=timeout)


def _parse_sync(path: str, filename: str, raw: bytes | None, *, timeout: int | None) -> dict:
    url = settings.mineru_url.rstrip("/") + "/file_parse"
    timeout = timeout or settings.mineru_timeout
    started = time.monotonic()
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout, connect=15)) as client:
            response = client.post(url, files=_files(path, filename), data=_params())
    except httpx.TimeoutException as exc:
        raise MinerUTimeout(f"MinerU request timed out after {timeout}s: {exc}") from exc
    except httpx.HTTPError as exc:
        raise MinerUError(f"MinerU request failed: {exc}") from exc
    _check_response_status(response)
    try:
        body = response.json()
    except ValueError as exc:
        raise MinerUError("MinerU returned a non-JSON response") from exc
    if isinstance(body, dict) and body.get("error"):
        raise MinerUError(f"MinerU error: {body['error']}")
    return _document_from_body(body, filename, raw, started)


def _parse_async(path: str, filename: str, raw: bytes | None, *, timeout: int | None,
                 progress=None) -> dict:
    base = settings.mineru_url.rstrip("/")
    timeout = timeout or settings.mineru_timeout
    started = time.monotonic()
    deadline = started + timeout
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=15, read=120, write=300, pool=30)) as client:
            response = client.post(base + "/tasks", files=_files(path, filename), data=_params())
            _check_response_status(response)
            submitted = response.json()
            task_id = submitted.get("task_id")
            if not task_id:
                raise MinerUError("MinerU did not return a task_id")
            # Never poll our own advertised URL: the service may announce a
            # host that is not reachable from here (container-internal names).
            status_url = f"{base}/tasks/{task_id}"
            logger.info("MinerU task submitted filename=%s task_id=%s queued_ahead=%s",
                        filename, task_id, submitted.get("queued_ahead"))

            delay = max(0.2, settings.mineru_poll_seconds)
            status = submitted.get("status") or "pending"
            while True:
                if time.monotonic() > deadline:
                    raise MinerUTimeout(
                        f"MinerU task {task_id} did not finish within {timeout}s")
                if status in {"failed", "error"}:
                    raise MinerUError(f"MinerU task {task_id} failed: {submitted.get('error')}")
                if status == "completed":
                    break
                if progress:
                    progress(submitted)
                time.sleep(delay)
                try:
                    status_response = client.get(status_url)
                except httpx.HTTPError as exc:
                    raise MinerUError(f"MinerU status check failed: {exc}") from exc
                if status_response.status_code != 200:
                    raise MinerUError(
                        f"MinerU status returned HTTP {status_response.status_code}")
                submitted = status_response.json()
                status = submitted.get("status") or ""

            result_response = client.get(f"{base}/tasks/{task_id}/result")
            _check_response_status(result_response)
            body = result_response.json()
    except (httpx.HTTPError, ValueError) as exc:
        if isinstance(exc, MinerUError):
            raise
        raise MinerUError(f"MinerU request failed: {exc}") from exc
    if isinstance(body, dict) and body.get("error"):
        raise MinerUError(f"MinerU error: {body['error']}")
    return _document_from_body(body, filename, raw, started)


def health() -> dict:
    """Reachability + queue status from the service's own `/health` endpoint."""
    if not mineru_available():
        return {"name": "MinerU", "status": "not_configured",
                "error": "MINERU_URL is empty"}
    base = settings.mineru_url.rstrip("/")
    started = time.monotonic()
    try:
        with httpx.Client(timeout=httpx.Timeout(10, connect=5)) as client:
            response = client.get(base + "/health")
        latency = int((time.monotonic() - started) * 1000)
        if response.status_code != 200:
            return {"name": "MinerU", "status": "error", "latency_ms": latency,
                    "error": f"HTTP {response.status_code}"}
        info = response.json() if response.content else {}
        queued = info.get("queued_tasks")
        processing = info.get("processing_tasks")
        detail = ""
        if queued is not None or processing is not None:
            detail = f"队列 {queued or 0} · 处理中 {processing or 0}"
        return {"name": "MinerU", "status": "connected", "latency_ms": latency,
                "endpoint": base + "/file_parse", "version": info.get("version", ""),
                "protocol_version": info.get("protocol_version"),
                "concurrency": info.get("max_concurrent_requests"),
                "queued_tasks": queued, "processing_tasks": processing,
                "detail": detail}
    except Exception as exc:
        return {"name": "MinerU", "status": "error",
                "latency_ms": int((time.monotonic() - started) * 1000),
                "error": str(exc)[:300]}
