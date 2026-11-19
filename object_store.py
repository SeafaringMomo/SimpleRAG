"""MinIO object storage adapter. MinIO client is imported lazily.

The endpoint and bucket come from `storage` so the settings page can repoint
them at runtime; the bucket is created when missing (`ensure_bucket`).
"""
from agent_test.rag.config import settings
from agent_test.rag.storage import ensure_bucket as _ensure_bucket, minio_client


def client():
    return minio_client()


def ensure_bucket():
    return _ensure_bucket()


def put_file(path: str, object_name: str, content_type: str):
    store = ensure_bucket()
    store.fput_object(settings.minio_bucket, object_name, path, content_type=content_type)
    return object_name


def remove_file(object_name: str):
    if object_name:
        client().remove_object(settings.minio_bucket, object_name)
