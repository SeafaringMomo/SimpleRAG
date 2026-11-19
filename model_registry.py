"""One OpenAI-compatible gateway for configured chat and embedding models."""
from __future__ import annotations

import time
import logging
import httpx
from openai import OpenAI
from agent_test.rag.config import settings
from agent_test.rag.usage_store import record_usage

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.client = OpenAI(base_url=settings.base_url, api_key=settings.api_key or "missing-key",
                             timeout=httpx.Timeout(120, connect=10), max_retries=1)

    def chat(self, model: str | None, messages: list[dict], task: str = "chat", **options):
        model = model or settings.chat_model
        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(model=model, messages=messages, **options)
            
            if not getattr(response, "choices", None):
                raise RuntimeError(
                    f"Chat API returned no choices for model={model}. "
                    f"Response: {response}"
                )
            usage = response.usage
            record_usage(model, task, getattr(usage, "prompt_tokens", 0) or 0,
                         getattr(usage, "completion_tokens", 0) or 0,
                         int((time.monotonic() - start) * 1000), True, usage is None)
            return response
        except Exception as exc:
            record_usage(model, task, latency_ms=int((time.monotonic() - start) * 1000), success=False, error=str(exc)[:1000])
            logger.exception("Model chat call failed model=%s task=%s", model, task)
            raise

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or settings.embedding_model
        start = time.monotonic()
        try:

            response = self.client.embeddings.create(model=model, input=texts)
            self._check_error_response(response, model)
            vectors = [list(item.embedding) for item in response.data]
            if len(vectors) != len(texts):
                raise RuntimeError(f"Embedding API returned {len(vectors)} vectors for {len(texts)} inputs")
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0
            record_usage(
                model, "embedding", prompt_tokens,
                max(0, total_tokens - prompt_tokens),
                int((time.monotonic() - start) * 1000), True, usage is None,
            )
            return vectors
        except Exception as exc:
            record_usage(
                model, "embedding",
                latency_ms=int((time.monotonic() - start) * 1000),
                success=False, error=str(exc)[:1000],
            )
            logger.exception("Model embedding call failed model=%s items=%d", model, len(texts))
            raise

    # ---------------------------------------------------------------- health
    def health(self):
        result = []
        for model in settings.models:
            start = time.monotonic()
            try:
                if model.api_style == "embedding":
                    self.embed(["health check"], model.name)
                elif model.category.lower() == "bge":
                    self.embed(["health check"], model.name)
                else:
                    self.chat(model.name, [{"role": "user", "content": "Reply OK"}], "health", max_tokens=8)
                result.append({
                    "name": model.name,
                    "category": model.category,
                    "status": "connected",
                    "latency_ms": int((time.monotonic() - start) * 1000),
                })
            except Exception as exc:
                result.append({
                    "name": model.name,
                    "category": model.category,
                    "status": "error",
                    "latency_ms": int((time.monotonic() - start) * 1000),
                    "error": str(exc)[:500],
                })
        return result

    @staticmethod
    def _check_error_response(response, model: str):
        """Validate gateway responses, including HTTP 200 business errors."""
        obj = getattr(response, "object", None)
        if obj == "error":
            raise RuntimeError(
                f"Embedding API error for model={model}: "
                f"[{getattr(response, 'code', None)}] {getattr(response, 'message', None)}"
            )
        data = getattr(response, "data", None)
        if not data:
            raise RuntimeError(f"Embedding API returned empty data for model={model}: {response}")
        for item in data:
            vector = getattr(item, "embedding", None)
            if not vector or not all(isinstance(x, (int, float)) for x in vector):
                raise RuntimeError(f"Embedding API returned an invalid vector for model={model}")


registry = ModelRegistry()
