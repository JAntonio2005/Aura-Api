from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from app.core.config import settings
from app.models.schemas import AssistantRequest, AssistantResponse
from app.services.assistant_context import (
    DEFAULT_DISCLAIMER,
    normalize_text,
    plain_text,
    suggested_followups,
)
from app.services.llm.base import AssistantEngine
from app.services.rag.retriever import RagRetriever


class RagOllamaEngine(AssistantEngine):
    name = "rag_ollama"

    def __init__(self):
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
        started = time.perf_counter()
        try:
            retrieved = self.retriever.retrieve_with_metadata(request.message)
            context = "\n\n".join(item.text for item in retrieved)
            prompt = self._build_prompt(request, breed_info, intent, context)
            answer = self._ask_ollama(prompt)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"[Assistant] RagOllama failed after {elapsed:.2f}s: {exc}")
            return self._error_response(
                request=request,
                breed_info=breed_info,
                intent=intent,
                safety_context=safety_context,
                message=f"No pude generar una respuesta RAG en este momento: {exc}",
            )

        answer = plain_text(answer)
        if len(answer) < 20:
            print("[Assistant] RagOllama response rejected: too short.")
            return self._error_response(
                request=request,
                breed_info=breed_info,
                intent=intent,
                safety_context=safety_context,
                message="No pude generar una respuesta RAG suficientemente clara en este momento.",
            )

        sources = self._sources(retrieved)
        for source in ("rag_ollama", "aura_assistant_docs"):
            if source not in sources:
                sources.append(source)
        print(f"[Assistant] RagOllama generated answer in {time.perf_counter() - started:.2f}s.")
        return AssistantResponse(
            answer=answer,
            intent=intent,
            breed=self._breed_out(breed_info),
            safety_level=safety_context.get("safety_level", "basic_guidance"),
            disclaimer=DEFAULT_DISCLAIMER if request.include_disclaimer else None,
            recommend_vet=bool(safety_context.get("recommend_vet")),
            sources=sources,
            suggested_followups=suggested_followups(intent, bool(safety_context.get("urgent"))),
        )

    def _build_prompt(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
        context: str,
    ) -> str:
        breed_name = plain_text((breed_info or {}).get("name")) if breed_info else "desconocida"
        dog_context = self._dog_context_text(request)
        clarification_rule = self._clarification_rule(request)
        return (
            "Eres Aura Assistant, un asistente de orientacion basica para cuidado canino.\n"
            "Reglas:\n"
            "- Responde en espanol, breve y claro.\n"
            "- Usa el contexto recuperado solo como informacion general.\n"
            "- No inventes edad, peso, diagnostico, sintomas, tiempo enfermo ni condicion fisica.\n"
            "- Si faltan datos especificos del perro, pide aclaracion en lugar de estimar.\n"
            "- Si el contexto menciona esperanza de vida de una raza, aclara que es expectativa general, no edad del perro.\n"
            "- No recomiendes medicamentos ni dosis.\n"
            "- Sugiere veterinario ante sintomas graves.\n"
            f"{clarification_rule}\n\n"
            f"Intencion: {intent}\n"
            f"Raza: {breed_name}\n"
            f"Datos especificos del perro proporcionados: {dog_context}\n"
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

    def _error_response(
        self,
        *,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
        safety_context: Dict[str, Any],
        message: str,
    ) -> AssistantResponse:
        return AssistantResponse(
            answer=message,
            intent=intent,
            breed=self._breed_out(breed_info),
            safety_level="rag_unavailable",
            disclaimer=DEFAULT_DISCLAIMER if request.include_disclaimer else None,
            recommend_vet=bool(safety_context.get("recommend_vet")),
            sources=["rag_ollama", "aura_assistant_docs"],
            suggested_followups=[
                "Verifica que Ollama este corriendo.",
                "Verifica que chroma_db exista y tenga la coleccion RAG.",
                "Ejecuta python -m app.services.rag.ingest si falta la base vectorial.",
            ],
        )

    def _breed_out(self, breed_info: Optional[Dict[str, Any]]):
        if not breed_info:
            return None

        from app.models.schemas import AssistantBreedOut

        return AssistantBreedOut(
            label=breed_info.get("slug") or breed_info.get("label"),
            name=plain_text(breed_info.get("name")),
        )

    def _sources(self, retrieved) -> list[str]:
        sources = []
        for item in retrieved:
            if item.source:
                source = item.source.replace("\\", "/")
                if source not in sources:
                    sources.append(source)
        return sources

    def _dog_context_text(self, request: AssistantRequest) -> str:
        if request.dog_context:
            parts = [
                f"{plain_text(key)}={plain_text(value)}"
                for key, value in request.dog_context.items()
                if value not in (None, "")
            ]
            if parts:
                return "; ".join(parts)
        return "no proporcionados"

    def _clarification_rule(self, request: AssistantRequest) -> str:
        text = normalize_text(request.message)
        dog_context = request.dog_context or {}
        context_text = normalize_text(" ".join(str(value) for value in dog_context.values()))
        asks_age = any(term in text for term in ("edad", "anos tiene", "meses tiene"))
        asks_weight = any(term in text for term in ("peso", "cuanto pesa", "kilos"))
        asks_symptoms = any(term in text for term in ("sintomas exactos", "que sintomas", "que tiene"))
        asks_duration = any(term in text for term in ("desde cuando", "cuanto tiempo", "tiempo enfermo"))
        asks_condition = any(term in text for term in ("condicion fisica", "estado fisico", "bajo peso"))
        asks_specific = asks_age or asks_weight or asks_symptoms or asks_duration or asks_condition

        if not asks_specific:
            return ""

        has_specific_data = bool(context_text)
        if asks_age:
            has_specific_data = has_specific_data or any(
                term in text or term in context_text
                for term in ("cachorro", "adulto", "senior", "viejo", "joven", "meses", "anos")
            )
        if asks_weight:
            has_specific_data = has_specific_data or any(
                term in text or term in context_text for term in ("kg", "kilo", "peso")
            )

        if has_specific_data:
            return ""

        if asks_age:
            return (
                "- La pregunta pide estimar edad, pero no hay datos especificos del perro. "
                "Responde que no tienes suficiente informacion para estimarla y pide datos como "
                "si es cachorro, adulto o senior, dientes, tamano y comportamiento."
            )
        if asks_weight:
            return (
                "- La pregunta pide peso, pero no hay datos especificos del perro. "
                "Pide tamano, condicion corporal o peso medido; no inventes kilos."
            )
        return (
            "- La pregunta pide datos especificos del perro, pero no fueron proporcionados. "
            "Pide aclaracion y no inventes informacion individual."
        )
