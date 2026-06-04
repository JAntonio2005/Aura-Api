from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.services.llm.base import AssistantEngine
from app.services.llm.picogpt_engine import PicoGptEngine
from app.services.llm.rag_ollama_engine import RagOllamaEngine
from app.services.llm.rules_engine import RulesEngine


@lru_cache(maxsize=4)
def get_assistant_engine(engine_name: str | None = None) -> AssistantEngine:
    requested = (engine_name or settings.ASSISTANT_ENGINE or "rules").strip().lower()
    fallback = RulesEngine()

    if requested == "picogpt":
        engine = PicoGptEngine(fallback=fallback)
        available, reason = engine.is_available()
        if available:
            print("[Assistant] Using engine: picogpt")
        else:
            print(f"[Assistant] Requested engine picogpt is not ready: {reason}. Fallback is enabled.")
        return engine

    if requested == "rag_ollama":
        engine = RagOllamaEngine(fallback=fallback)
        available, reason = engine.is_available()
        if available:
            print("[Assistant] Using engine: rag_ollama")
        else:
            print(f"[Assistant] Requested engine rag_ollama is not ready: {reason}. Fallback is enabled.")
        return engine

    if requested != "rules":
        print(f"[Assistant] Unknown ASSISTANT_ENGINE={requested!r}; using rules.")

    print("[Assistant] Using engine: rules")
    return fallback
