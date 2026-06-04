from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings
from app.services.rag.vector_store import collection_status, rag_chroma_path


@dataclass(frozen=True)
class RetrievedContext:
    text: str
    source: str | None = None
    metadata: dict | None = None


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("intfloat/multilingual-e5-small")


@lru_cache(maxsize=4)
def get_chroma_collection(collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(rag_chroma_path()))
    return client.get_collection(collection_name)


class RagRetriever:
    def __init__(self, *, collection_name: str | None = None, top_k: int | None = None):
        self.collection_name = collection_name or settings.RAG_COLLECTION
        self.top_k = top_k or settings.RAG_TOP_K

    def is_available(self) -> tuple[bool, str]:
        return collection_status()

    def retrieve(self, query: str) -> list[str]:
        return [item.text for item in self.retrieve_with_metadata(query)]

    def retrieve_with_metadata(self, query: str) -> list[RetrievedContext]:
        available, reason = self.is_available()
        if not available:
            raise RuntimeError(reason)

        model = get_embedding_model()
        embedding = model.encode(query, normalize_embeddings=True).tolist()

        collection = get_chroma_collection(self.collection_name)
        results = collection.query(query_embeddings=[embedding], n_results=self.top_k)
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        retrieved = []
        for document, metadata in zip(documents, metadatas):
            metadata = metadata or {}
            retrieved.append(
                RetrievedContext(
                    text=document,
                    source=metadata.get("source"),
                    metadata=metadata,
                )
            )
        return retrieved
