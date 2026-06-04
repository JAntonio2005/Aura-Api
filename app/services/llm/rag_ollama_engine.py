from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from app.core.config import settings
from app.models.schemas import AssistantRequest, AssistantResponse
from app.services.llm.base import AssistantEngine
from app.services.llm.rules_engine import RulesEngine, plain_text
from app.services.rag.retriever import RagRetriever


class RagOllamaEngine(AssistantEngine):
    name = "rag_ollama"

    def __init__(self, fallback: Optional[AssistantEngine] = None):
        self.fallback = fallback or RulesEngine()
        self.retriever = RagRetriever()

    def is_available(self) -> tuple[bool, str]:
        return self.retriever.is_available()

    def generate(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
        safety_context: Dict[str, Any],
    ) -> AssistantResponse:
        base_response = self.fallback.generate(request, breed_info, intent, safety_context)

        if safety_context.get("urgent") or base_response.recommend_vet:
            print("[Assistant] RagOllama bypassed: urgent safety context; using RulesEngine.")
            return base_response

        available, reason = self.is_available()
        if not available:
            print(f"[Assistant] RagOllama unavailable: {reason}; using RulesEngine.")
            return base_response

        started = time.perf_counter()
        try:
            context = "\n\n".join(self.retriever.retrieve(request.message))
            prompt = self._build_prompt(request, breed_info, intent, context)
            answer = self._ask_ollama(prompt)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"[Assistant] RagOllama failed after {elapsed:.2f}s: {exc}; using RulesEngine.")
            return base_response

        answer = plain_text(answer)
        if len(answer) < 20:
            print("[Assistant] RagOllama response rejected: too short; using RulesEngine.")
            return base_response

        response_data = base_response.model_dump()
        response_data["answer"] = answer
        sources = list(response_data.get("sources") or [])
        for source in ("rag_ollama", "aura_assistant_docs"):
            if source not in sources:
                sources.append(source)
        response_data["sources"] = sources
        print(f"[Assistant] RagOllama generated answer in {time.perf_counter() - started:.2f}s.")
        return AssistantResponse(**response_data)

    def _build_prompt(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
        context: str,
    ) -> str:
        breed_name = plain_text((breed_info or {}).get("name")) if breed_info else "desconocida"
        return (
            "Eres Aura Assistant, un asistente de orientacion basica para cuidado canino.\n"
            "Reglas: responde en espanol, usa el contexto, no inventes, no recomiendes "
            "medicamentos ni dosis, y sugiere veterinario ante sintomas graves.\n\n"
            f"Intencion: {intent}\n"
            f"Raza: {breed_name}\n"
            f"Contexto recuperado:\n{context}\n\n"
            f"Pregunta: {request.message}\n"
            "Respuesta breve y segura:"
        )

    def _ask_ollama(self, prompt: str) -> str:
        url = settings.OLLAMA_BASE_URL.rstrip("/") + "/api/generate"
        payload = json.dumps(
            {"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=settings.OLLAMA_TIMEOUT_SECONDS,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        return str(data.get("response", ""))
