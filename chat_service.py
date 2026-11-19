"""RAG retrieval and answer generation."""
from __future__ import annotations

from agent_test.rag.config import settings
from agent_test.rag.model_registry import registry
from agent_test.rag.usage_store import connection
from agent_test.rag.embedding_service import ensure_collection


def retrieve(query: str, top_k: int = 5, embedding_model: str | None = None,
             collection_name: str | None = None) -> list[dict]:
    vectors = registry.embed([query], embedding_model)
    from agent_test.rag.storage import milvus_client
    client = milvus_client()
    collection_name = collection_name or settings.collection
    selected_embedding_model = embedding_model or settings.embedding_model
    with connection() as db:
        stored = {row[0] for row in db.execute(
            "SELECT DISTINCT embedding_model FROM documents WHERE status='ready' AND collection_name=? AND embedding_model<>''",
            (collection_name,),
        )}
    if stored and stored != {selected_embedding_model}:
        raise ValueError(
            f"Collection {collection_name} contains vectors from embedding model(s) "
            f"{', '.join(sorted(stored))}; selected model is {selected_embedding_model}. "
            "Rebuild the knowledge base with the selected model or choose the model used to index it."
        )
    if not client.has_collection(collection_name):
        return []
    ensure_collection(client, len(vectors[0]), collection_name)
    results = client.search(collection_name=collection_name, data=vectors, limit=top_k,
                            output_fields=["doc_id", "doc_name", "content", "chunk_index"],
                            search_params={"metric_type": "COSINE", "params": {}})
    return [{"id": hit["id"], "score": hit.get("distance", 0), **hit.get("entity", {})} for hit in results[0]]


def answer(query: str, top_k: int = 5, temperature: float | None = None,
           max_tokens: int | None = None, chat_model: str | None = None,
           embedding_model: str | None = None, collection_name: str | None = None) -> dict:
    hits = retrieve(query, top_k, embedding_model, collection_name)
    if not hits:
        return {"answer": "知识库中还没有可用的相关内容，请先上传并处理文档。", "sources": []}
    context = "\n\n".join(f"[来源 {i+1}: {h.get('doc_name')} / 片段 {h.get('chunk_index')} ]\n{h.get('content')}" for i, h in enumerate(hits))
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    response = registry.chat(chat_model or settings.chat_model, [
        {"role": "system", "content": "你是知识库问答助手。优先依据提供的资料回答；资料不足时明确说明。回答中用[来源 N]标出依据。"},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}"}], task="rag_chat", **options)
    return {"answer": response.choices[0].message.content or "", "sources": hits}
