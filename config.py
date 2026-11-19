"""Environment backed configuration for the single user RAG workbench."""
from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelConfig:
    '''
    name: str,模型名称
    category: str，模型厂商
    api_style: str = "chat"，模型类别
    
    '''
    name: str
    category: str
    api_style: str = "chat"


@dataclass(frozen=True)
class Settings:
    base_url: str = os.getenv("AI_BASE_URL", "")
    api_key: str = os.getenv("AI_API_KEY", "")
    chat_model: str = os.getenv("CHAT_MODEL", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "0"))
    milvus_uri: str = os.getenv("MILVUS_URI", "")
    milvus_token: str = os.getenv("MILVUS_TOKEN", "")
    milvus_database: str = os.getenv("MILVUS_DATABASE", "")
    collection: str = os.getenv("MILVUS_COLLECTION", "tmp_documents")
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "test-documents")
    minio_secure: bool = _bool("MINIO_SECURE", False)
    # Web console ports of the two storage services (Milvus: 9091, MinIO: 9001).
    milvus_console_port: int = int(os.getenv("MILVUS_CONSOLE_PORT", "9091"))
    minio_console_port: int = int(os.getenv("MINIO_CONSOLE_PORT", "9001"))
    database_path: str = os.getenv("USAGE_DATABASE", "data/usage.sqlite3")
    log_path: str = os.getenv("LOG_FILE", "logs/app.log")
    log_max_bytes: int = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))
    log_backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    upload_dir: str = os.getenv("UPLOAD_TEMP_DIR", "data/uploads")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "100"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    health_cache_seconds: int = int(os.getenv("HEALTH_CACHE_SECONDS", "60"))
    # ------------------------------------------------------------- MinerU
    mineru_url: str = os.getenv("MINERU_URL", "")
    mineru_backend: str = os.getenv("MINERU_BACKEND", "pipeline")
    mineru_parse_method: str = os.getenv("MINERU_PARSE_METHOD", "auto")
    mineru_lang: str = os.getenv("MINERU_LANG", "ch")
    mineru_formula_enable: bool = _bool("MINERU_FORMULA_ENABLE", True)
    mineru_table_enable: bool = _bool("MINERU_TABLE_ENABLE", True)
    mineru_timeout: int = int(os.getenv("MINERU_TIMEOUT", "900"))
    # MinerU 3.x hybrid backends take an effort level; pipeline ignores it.
    mineru_effort: str = os.getenv("MINERU_EFFORT", "medium")
    mineru_image_analysis: bool = _bool("MINERU_IMAGE_ANALYSIS", True)
    # Submit through /tasks and poll, so long parses do not hold one HTTP
    # connection open for the whole job and progress can be surfaced.
    mineru_async: bool = _bool("MINERU_ASYNC", True)
    mineru_poll_seconds: float = float(os.getenv("MINERU_POLL_SECONDS", "2"))
    # When MinerU is unreachable, fall back to the local MarkItDown converter.
    mineru_fallback: bool = _bool("MINERU_FALLBACK_TO_MARKITDOWN", True)
    # Extensions routed to MinerU; everything else uses a local converter.
    mineru_extensions: str = os.getenv(
        "MINERU_EXTENSIONS",
        ".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.gif,.webp,.docx,.pptx,.xlsx")
    # How many conversions run at once. MinerU's own worker pool is small
    # (max_concurrent_requests), so more local parallelism only queues upstream.
    conversion_workers: int = int(os.getenv("CONVERSION_WORKERS", "2"))

    @property
    def models(self) -> list[ModelConfig]:
        raw = os.getenv("MODELS", "")
        if raw:
            models = []
            for item in raw.split(","):
                bits = item.strip().split(":")
                if len(bits) >= 2:
                    models.append(ModelConfig(bits[0], bits[1], bits[2] if len(bits) > 2 else "chat"))
            if models:
                return models
        #使用默认值    
        return [ModelConfig(self.chat_model, "deepseek", "chat"), ModelConfig(self.embedding_model, "bge", "embedding")]


settings = Settings()
