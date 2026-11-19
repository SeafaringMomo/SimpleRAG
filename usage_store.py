"""Small local SQLite ledger for API usage and health-check history."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from agent_test.rag.config import settings


@contextmanager
def connection():
    os.makedirs(os.path.dirname(settings.database_path) or ".", exist_ok=True)
    db = sqlite3.connect(settings.database_path)
    db.row_factory = sqlite3.Row
    try:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            model TEXT NOT NULL, task TEXT NOT NULL, input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0,
            latency_ms INTEGER NOT NULL DEFAULT 0, success INTEGER NOT NULL,
            estimated INTEGER NOT NULL DEFAULT 0, error TEXT
        );
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, object_key TEXT, sha256 TEXT NOT NULL,
            status TEXT NOT NULL, chunks INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
            error TEXT, chunk_size INTEGER NOT NULL DEFAULT 900,
            chunk_overlap INTEGER NOT NULL DEFAULT 120,
            embedding_model TEXT NOT NULL DEFAULT '', collection_name TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ingest_tasks (
            id TEXT PRIMARY KEY, filename TEXT NOT NULL, status TEXT NOT NULL,
            current_step TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            details TEXT NOT NULL DEFAULT '[]', error TEXT, document_id TEXT
        );
        CREATE TABLE IF NOT EXISTS conversions (
            id TEXT PRIMARY KEY, batch_id TEXT NOT NULL DEFAULT '',
            filename TEXT NOT NULL, status TEXT NOT NULL,
            stage TEXT NOT NULL DEFAULT '', message TEXT NOT NULL DEFAULT '',
            converter TEXT NOT NULL DEFAULT '', object_key TEXT NOT NULL DEFAULT '',
            json_key TEXT NOT NULL DEFAULT '', md_key TEXT NOT NULL DEFAULT '',
            sha256 TEXT NOT NULL DEFAULT '', page_count INTEGER NOT NULL DEFAULT 0,
            block_count INTEGER NOT NULL DEFAULT 0, char_count INTEGER NOT NULL DEFAULT 0,
            image_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            error TEXT, document_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_conversions_batch ON conversions(batch_id);
        """)
        columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
        if "chunk_size" not in columns:
            db.execute("ALTER TABLE documents ADD COLUMN chunk_size INTEGER NOT NULL DEFAULT 900")
        if "chunk_overlap" not in columns:
            db.execute("ALTER TABLE documents ADD COLUMN chunk_overlap INTEGER NOT NULL DEFAULT 120")
        if "embedding_model" not in columns:
            db.execute("ALTER TABLE documents ADD COLUMN embedding_model TEXT NOT NULL DEFAULT ''")
        if "collection_name" not in columns:
            db.execute("ALTER TABLE documents ADD COLUMN collection_name TEXT NOT NULL DEFAULT ''")
        yield db
        db.commit()
    finally:
        db.close()


def init_db():
    with connection():
        pass


def record_usage(model: str, task: str, input_tokens: int = 0, output_tokens: int = 0,
                 latency_ms: int = 0, success: bool = True, estimated: bool = False, error: str | None = None):
    with connection() as db:
        db.execute("INSERT INTO usage(created_at,model,task,input_tokens,output_tokens,total_tokens,latency_ms,success,estimated,error) VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (datetime.now(timezone.utc).isoformat(), model, task, input_tokens, output_tokens,
                    input_tokens + output_tokens, latency_ms, int(success), int(estimated), error))


def usage_summary():
    with connection() as db:
        totals = dict(db.execute("SELECT COUNT(*) calls, COALESCE(SUM(success),0) successes, COALESCE(SUM(input_tokens),0) input_tokens, COALESCE(SUM(output_tokens),0) output_tokens, COALESCE(SUM(total_tokens),0) total_tokens FROM usage").fetchone())
        totals["models"] = [dict(r) for r in db.execute("SELECT model, COUNT(*) calls, COALESCE(SUM(total_tokens),0) total_tokens, COALESCE(SUM(success),0) successes FROM usage GROUP BY model ORDER BY model")]
        totals["recent"] = [dict(r) for r in db.execute("SELECT created_at,model,task,input_tokens,output_tokens,total_tokens,latency_ms,success,estimated,error FROM usage ORDER BY id DESC LIMIT 30")]
        return totals


def list_documents():
    with connection() as db:
        return [dict(r) for r in db.execute("SELECT * FROM documents ORDER BY created_at DESC")]


def get_setting(key: str, default: str) -> str:
    with connection() as db:
        row = db.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_settings(values: dict[str, str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as db:
        db.executemany("INSERT INTO app_settings(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                       [(key, value, now) for key, value in values.items()])


def delete_settings(keys: list[str]) -> None:
    with connection() as db:
        db.executemany("DELETE FROM app_settings WHERE key=?", [(key,) for key in keys])


def selected_models() -> tuple[str, str]:
    return get_setting("chat_model", settings.chat_model), get_setting("embedding_model", settings.embedding_model)


def create_ingest_task(task_id: str, filename: str) -> None:
    import json
    now = datetime.now(timezone.utc).isoformat()
    details = [{"key": step, "label": label, "status": "pending", "message": "等待处理"}
               for step, label in [
                   ("upload", "文档上传"), ("minio", "保存原文件到 MinIO"),
                   ("split", "文本提取与切分"), ("embedding", "调用 BGE 生成向量"),
                   ("collection", "检查/创建并加载 Milvus collection"),
                   ("persist", "保存切分文本与向量到 Milvus")]]
    with connection() as db:
        db.execute("INSERT INTO ingest_tasks(id,filename,status,current_step,created_at,updated_at,details) VALUES(?,?,?,?,?,?,?)",
                   (task_id, filename, "processing", "upload", now, now, json.dumps(details, ensure_ascii=False)))


def update_ingest_task(task_id: str, step: str, status: str, message: str = "", *,
                       task_status: str | None = None, error: str | None = None,
                       document_id: str | None = None) -> None:
    import json
    now = datetime.now(timezone.utc).isoformat()
    with connection() as db:
        row = db.execute("SELECT details,status FROM ingest_tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            return
        details = json.loads(row["details"])
        for item in details:
            if item["key"] == step:
                item.update(status=status, message=message)
                break
        db.execute("UPDATE ingest_tasks SET details=?,current_step=?,status=?,updated_at=?,error=COALESCE(?,error),document_id=COALESCE(?,document_id) WHERE id=?",
                   (json.dumps(details, ensure_ascii=False), step,
                    task_status or row["status"], now, error, document_id, task_id))


def get_ingest_task(task_id: str) -> dict | None:
    import json
    with connection() as db:
        row = db.execute("SELECT * FROM ingest_tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        return None
    result = dict(row)
    result["details"] = json.loads(result["details"])
    return result


def recent_ingest_tasks(limit: int = 20) -> list[dict]:
    import json
    with connection() as db:
        rows = db.execute("SELECT * FROM ingest_tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["details"] = json.loads(item["details"])
        result.append(item)
    return result


# ------------------------------------------------------------- conversions
def create_conversion(conversion_id: str, batch_id: str, filename: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as db:
        db.execute(
            "INSERT INTO conversions(id,batch_id,filename,status,stage,message,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (conversion_id, batch_id, filename, "processing", "received", "等待解析", now, now))


def update_conversion(conversion_id: str, *, status: str | None = None,
                      stage: str | None = None, message: str | None = None,
                      error: str | None = None, document_id: str | None = None,
                      **fields) -> None:
    allowed = {"converter", "object_key", "json_key", "md_key", "sha256",
               "page_count", "block_count", "char_count", "image_count"}
    updates = {key: value for key, value in fields.items() if key in allowed}
    if status is not None:
        updates["status"] = status
    if stage is not None:
        updates["stage"] = stage
    if message is not None:
        updates["message"] = message
    if error is not None:
        updates["error"] = error
    if document_id is not None:
        updates["document_id"] = document_id
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    columns = ", ".join(f"{key}=?" for key in updates)
    with connection() as db:
        db.execute(f"UPDATE conversions SET {columns} WHERE id=?",
                   (*updates.values(), conversion_id))


def get_conversion(conversion_id: str) -> dict | None:
    with connection() as db:
        row = db.execute("SELECT * FROM conversions WHERE id=?", (conversion_id,)).fetchone()
    return dict(row) if row else None


def list_conversions(limit: int = 50, batch_id: str | None = None) -> list[dict]:
    with connection() as db:
        if batch_id:
            rows = db.execute(
                "SELECT * FROM conversions WHERE batch_id=? ORDER BY created_at DESC",
                (batch_id,)).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM conversions ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
    return [dict(row) for row in rows]


def delete_conversion(conversion_id: str) -> None:
    with connection() as db:
        db.execute("DELETE FROM conversions WHERE id=?", (conversion_id,))


def claim_conversion(conversion_id: str, *, status: str = "processing",
                     stage: str = "received", message: str = "") -> bool:
    """Atomically move a conversion out of an idle state.

    Returns False when another request (or a running job) already owns it, so a
    double-clicked retry cannot start two parses of the same record.
    """
    now = datetime.now(timezone.utc).isoformat()
    with connection() as db:
        cursor = db.execute(
            "UPDATE conversions SET status=?,stage=?,message=?,error='',updated_at=? "
            "WHERE id=? AND status NOT IN ('processing','indexing')",
            (status, stage, message, now, conversion_id))
        return cursor.rowcount > 0
