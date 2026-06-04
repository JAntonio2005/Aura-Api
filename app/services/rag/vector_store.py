from __future__ import annotations

import importlib.util
from pathlib import Path

from app.core.config import PROJECT_ROOT, settings


def rag_chroma_path() -> Path:
    path = Path(settings.RAG_CHROMA_DIR)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def dependency_status() -> tuple[bool, str]:
    missing = [
        module
        for module in ("chromadb", "sentence_transformers")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return False, "missing Python modules: " + ", ".join(missing)
    return True, "ready"


def collection_status() -> tuple[bool, str]:
    ready, reason = dependency_status()
    if not ready:
        return False, reason

    chroma_dir = rag_chroma_path()
    if not chroma_dir.exists():
        return False, f"Chroma directory not found: {chroma_dir}"

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_dir))
        client.get_collection(settings.RAG_COLLECTION)
    except Exception as exc:
        return False, f"Chroma collection {settings.RAG_COLLECTION!r} is not ready: {exc}"

    return True, "ready"
