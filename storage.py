"""Effective MinIO / Milvus endpoints with runtime overrides.

Endpoints start from `.env` (MINIO_ENDPOINT, MILVUS_URI) but can be changed from
the 「存储设置」 page without editing files; overrides live in the SQLite
`app_settings` table. Buckets, databases and collections are created on demand,
so pointing the app at an empty server is enough to get started.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from agent_test.rag.config import settings
from agent_test.rag.usage_store import get_setting, set_settings

logger = logging.getLogger(__name__)

MILVUS_HOST_KEY = "storage.milvus.host"
MILVUS_PORT_KEY = "storage.milvus.port"
MINIO_HOST_KEY = "storage.minio.host"
MINIO_PORT_KEY = "storage.minio.port"

# Milvus rejects anything but letters, digits and underscores.
_INVALID_DB_CHARS = re.compile(r"[^0-9A-Za-z_]")


def _split_endpoint(value: str, default_port: int) -> tuple[str, int]:
    """Accept `host`, `host:port`, `http://host:port` and return (host, port)."""
    raw = (value or "").strip()
    if not raw:
        return "", default_port
    if "://" not in raw:
        raw = f"//{raw}"
    parsed = urlparse(raw)
    host = parsed.hostname or ""
    try:
        port = parsed.port or default_port
    except ValueError:
        port = default_port
    return host, port


def _milvus_defaults() -> tuple[str, int]:
    return _split_endpoint(settings.milvus_uri, 19530)


def _minio_defaults() -> tuple[str, int]:
    return _split_endpoint(settings.minio_endpoint, 9000)


def milvus_host() -> str:
    host, _ = _milvus_defaults()
    return get_setting(MILVUS_HOST_KEY, host)


def milvus_port() -> int:
    _, port = _milvus_defaults()
    try:
        return int(get_setting(MILVUS_PORT_KEY, str(port)))
    except ValueError:
        return port


def minio_host() -> str:
    host, _ = _minio_defaults()
    return get_setting(MINIO_HOST_KEY, host)


def minio_port() -> int:
    _, port = _minio_defaults()
    try:
        return int(get_setting(MINIO_PORT_KEY, str(port)))
    except ValueError:
        return port


def milvus_uri() -> str:
    return f"http://{milvus_host()}:{milvus_port()}"


def minio_endpoint() -> str:
    return f"{minio_host()}:{minio_port()}"


def milvus_database_name() -> str:
    """The configured Milvus database, sanitised to a legal name ('' = default)."""
    raw = (settings.milvus_database or "").strip()
    if not raw:
        return ""
    cleaned = _INVALID_DB_CHARS.sub("_", raw)
    if cleaned != raw:
        logger.warning("MILVUS_DATABASE=%r is not a legal Milvus name; using %r instead",
                       raw, cleaned)
    return cleaned


def milvus_console_url() -> str:
    return f"http://{milvus_host()}:{settings.milvus_console_port}"


def minio_console_url() -> str:
    scheme = "https" if settings.minio_secure else "http"
    return f"{scheme}://{minio_host()}:{settings.minio_console_port}"


def current() -> dict:
    """Everything the settings page needs to render."""
    return {
        "milvus": {"host": milvus_host(), "port": milvus_port(),
                   "uri": milvus_uri(), "database": milvus_database_name(),
                   "collection": settings.collection,
                   "console_url": milvus_console_url(),
                   "console_port": settings.milvus_console_port},
        "minio": {"host": minio_host(), "port": minio_port(),
                  "endpoint": minio_endpoint(), "bucket": settings.minio_bucket,
                  "console_url": minio_console_url(),
                  "console_port": settings.minio_console_port,
                  "secure": settings.minio_secure},
        "env": {"milvus_uri": settings.milvus_uri, "minio_endpoint": settings.minio_endpoint},
    }


def save(milvus_host_value: str, milvus_port_value: int,
         minio_host_value: str, minio_port_value: int) -> dict:
    set_settings({
        MILVUS_HOST_KEY: milvus_host_value.strip(),
        MILVUS_PORT_KEY: str(milvus_port_value),
        MINIO_HOST_KEY: minio_host_value.strip(),
        MINIO_PORT_KEY: str(minio_port_value),
    })
    logger.info("Storage endpoints updated milvus=%s:%s minio=%s:%s",
                milvus_host_value, milvus_port_value, minio_host_value, minio_port_value)
    return current()


# --------------------------------------------------------------- Milvus
def milvus_client(db_name: str | None = None, *, ensure_database: bool = True,
                  timeout: float | None = None):
    """Milvus client whose database exists (created on first use if missing)."""
    from pymilvus import MilvusClient

    target = milvus_database_name() if db_name is None else db_name
    extra = {"timeout": timeout} if timeout is not None else {}
    client = MilvusClient(uri=milvus_uri(), token=settings.milvus_token or None,
                          db_name=target or "default", **extra)
    if target and ensure_database:
        try:
            client.list_collections()
        except Exception as exc:
            # Database is absent (or the server just came up): create it, then retry once.
            logger.warning("Milvus database %r unavailable (%s); attempting to create it",
                           target, exc)
            root = MilvusClient(uri=milvus_uri(), token=settings.milvus_token or None, **extra)
            try:
                root.create_database(target, **extra)
                logger.info("Milvus database created name=%s", target)
            except Exception as create_exc:
                logger.warning("Milvus database creation skipped name=%s error=%s",
                               target, create_exc)
            client = MilvusClient(uri=milvus_uri(), token=settings.milvus_token or None,
                                  db_name=target, **extra)
    return client


# --------------------------------------------------------------- MinIO
def _http_client(connect: float = 5, read: float = 15, retries: int = 2):
    """urllib3 pool with bounded timeouts; MinIO's default retries hang for minutes."""
    import urllib3
    return urllib3.PoolManager(
        timeout=urllib3.util.Timeout(connect=connect, read=read),
        retries=urllib3.util.Retry(total=retries, backoff_factor=0.2,
                                   status_forcelist=[500, 502, 503, 504]))


def minio_client(probe: bool = False):
    from minio import Minio
    http_client = _http_client(connect=3, read=5, retries=0) if probe else None
    # MinIO only fails fast with a bounded pool; the default retries for minutes.
    kwargs = {"http_client": http_client} if http_client is not None else {}
    return Minio(minio_endpoint(), access_key=settings.minio_access_key,
                 secret_key=settings.minio_secret_key, secure=settings.minio_secure, **kwargs)


def ensure_bucket(store=None, probe: bool = False):
    """Return a MinIO client whose configured bucket exists."""
    store = store or minio_client(probe=probe)
    if not store.bucket_exists(settings.minio_bucket):
        store.make_bucket(settings.minio_bucket)
        logger.info("MinIO bucket created name=%s endpoint=%s",
                    settings.minio_bucket, minio_endpoint())
    return store


def probe() -> dict:
    """Live connectivity check used by the settings page."""
    result: dict = {}
    try:
        client = milvus_client(timeout=5)
        collections = client.list_collections()
        result["milvus"] = {"status": "connected", "uri": milvus_uri(),
                            "database": milvus_database_name() or "default",
                            "collections": list(collections)}
    except Exception as exc:
        result["milvus"] = {"status": "error", "uri": milvus_uri(), "error": str(exc)[:300]}
    try:
        store = ensure_bucket(probe=True)
        buckets = [b.name for b in store.list_buckets()]
        result["minio"] = {"status": "connected", "endpoint": minio_endpoint(),
                           "bucket": settings.minio_bucket, "buckets": buckets}
    except Exception as exc:
        result["minio"] = {"status": "error", "endpoint": minio_endpoint(),
                           "error": str(exc)[:300]}
    return result
