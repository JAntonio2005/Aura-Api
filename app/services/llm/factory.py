from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.services.llm.base import AssistantEngine
from app.services.llm.hf_rag_engine import HfRagEngine
from app.services.llm.rag_ollama_engine import RagOllamaEngine
from app.services.llm.rules_engine import RulesEngine


@lru_cache(maxsize=4)
def get_assistant_engine(engine_name: str | None = None) -> AssistantEngine:
    requested = (engine_name or settings.ASSISTANT_ENGINE or "rag_ollama").strip().lower()

    if requested == "hf_rag":
        engine = HfRagEngine()
        available, reason = engine.is_available()
        if available:
            print("[Assistant] Using engine: hf_rag")
        else:
            print(f"[Assistant] Using engine: hf_rag, but it is not ready: {reason}.")
        return engine

    if requested == "rag_ollama":
        engine = RagOllamaEngine()
        available, reason = engine.is_available()
        if available:
            print("[Assistant] Using engine: rag_ollama")
        else:
            print(f"[Assistant] Using engine: rag_ollama, but RAG is not ready: {reason}.")
        return engine

    if requested == "rules":
        print("[Assistant] Using legacy engine: rules")
        return RulesEngine()

    if requested == "picogpt":
        print("[Assistant] ASSISTANT_ENGINE=picogpt is legacy and not used by the default RAG flow.")
        return RagOllamaEngine()

    print(f"[Assistant] Unknown ASSISTANT_ENGINE={requested!r}; using rag_ollama.")
    return RagOllamaEngine()
