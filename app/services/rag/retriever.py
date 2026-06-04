from __future__ import annotations

from app.core.config import settings
from app.services.rag.vector_store import collection_status, rag_chroma_path


class RagRetriever:
    def __init__(self, *, collection_name: str | None = None, top_k: int | None = None):
        self.collection_name = collection_name or settings.RAG_COLLECTION
        self.top_k = top_k or settings.RAG_TOP_K

    def is_available(self) -> tuple[bool, str]:
        return collection_status()

    def retrieve(self, query: str) -> list[str]:
        available, reason = self.is_available()
        if not available:
            raise RuntimeError(reason)

        import chromadb
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("intfloat/multilingual-e5-small")
        embedding = model.encode(query, normalize_embeddings=True).tolist()

        client = chromadb.PersistentClient(path=str(rag_chroma_path()))
        collection = client.get_collection(self.collection_name)
        results = collection.query(query_embeddings=[embedding], n_results=self.top_k)
        return list((results.get("documents") or [[]])[0])
