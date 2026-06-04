from __future__ import annotations

import json
from pathlib import Path

from app.core.config import PROJECT_ROOT, settings
from app.services.rag.chunker import chunk_text, iter_supported_documents
from app.services.rag.vector_store import dependency_status, rag_chroma_path


DOCS_DIR = PROJECT_ROOT / "data" / "assistant_docs"
BREEDS_JSON = PROJECT_ROOT / "app" / "data" / "breeds.json"


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False)
    return path.read_text(encoding="utf-8")


def build_local_chunks() -> list[dict]:
    chunks = []
    for path in iter_supported_documents(DOCS_DIR) or []:
        for index, chunk in enumerate(chunk_text(_read_document(path))):
            chunks.append({"id": f"{path.stem}_{index}", "text": chunk, "source": str(path)})

    if BREEDS_JSON.exists():
        breed_data = json.loads(BREEDS_JSON.read_text(encoding="utf-8"))
        for index, chunk in enumerate(chunk_text(json.dumps(breed_data, ensure_ascii=False))):
            chunks.append({"id": f"breeds_{index}", "text": chunk, "source": str(BREEDS_JSON)})

    return chunks


def ingest() -> int:
    ready, reason = dependency_status()
    if not ready:
        raise RuntimeError(reason)

    import chromadb
    from sentence_transformers import SentenceTransformer

    chunks = build_local_chunks()
    client = chromadb.PersistentClient(path=str(rag_chroma_path()))
    try:
        client.delete_collection(settings.RAG_COLLECTION)
    except Exception:
        pass
    collection = client.get_or_create_collection(settings.RAG_COLLECTION)
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    for chunk in chunks:
        embedding = model.encode(chunk["text"], normalize_embeddings=True).tolist()
        collection.add(
            ids=[chunk["id"]],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{"source": chunk["source"]}],
        )
    return len(chunks)


if __name__ == "__main__":
    total = ingest()
    print(f"Indexed chunks: {total}")
