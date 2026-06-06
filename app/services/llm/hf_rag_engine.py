from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from app.core.config import settings
from app.models.schemas import AssistantBreedOut, AssistantRequest, AssistantResponse
from app.services.assistant_context import DEFAULT_DISCLAIMER, plain_text, suggested_followups
from app.services.llm.base import AssistantEngine
from app.services.rag.simple_retriever import SimpleRetriever


class HfRagEngine(AssistantEngine):
    name = "hf_rag"

    def __init__(self):
        self.retriever = SimpleRetriever(top_k=settings.HF_RAG_TOP_K)

    def is_available(self) -> tuple[bool, str]:
        docs_available, docs_reason = self.retriever.is_available()
        if not docs_available:
            return False, docs_reason
        if not settings.HF_TOKEN:
            return False, "HF_TOKEN is not configured"
        if not settings.HF_MODEL:
            return False, "HF_MODEL is not configured"
        return True, "ready"

    def generate(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
        safety_context: Dict[str, Any],
    ) -> AssistantResponse:
        started = time.perf_counter()
        try:
            retrieved = self.retriever.retrieve_with_metadata(
                self._retrieval_query(request, breed_info, intent),
                top_k=settings.HF_RAG_TOP_K,
            )
            context = "\n\n".join(f"[{item.source}]\n{item.text}" for item in retrieved)
            messages = self._build_messages(request, breed_info, intent, safety_context, context)
            answer = self._ask_hugging_face(messages)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"[Assistant] HfRag failed after {elapsed:.2f}s: {self._safe_error(exc)}")
            return self._error_response(
                request=request,
                breed_info=breed_info,
                intent=intent,
                safety_context=safety_context,
                message=(
                    "No pude generar una respuesta con Hugging Face en este momento. "
                    "Verifica la configuracion del servicio e intenta de nuevo."
                ),
            )

        answer = plain_text(answer)
        if len(answer) < 20:
            print("[Assistant] HfRag response rejected: too short.")
            return self._error_response(
                request=request,
                breed_info=breed_info,
                intent=intent,
                safety_context=safety_context,
                message="No pude generar una respuesta suficientemente clara en este momento.",
            )

        sources = self._sources(retrieved)
        for source in ("hf_rag", "aura_assistant_docs"):
            if source not in sources:
                sources.append(source)

        print(f"[Assistant] HfRag generated answer in {time.perf_counter() - started:.2f}s.")
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

    def _build_messages(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
        safety_context: Dict[str, Any],
        context: str,
    ) -> list[dict[str, str]]:
        breed_name = plain_text((breed_info or {}).get("name")) if breed_info else "desconocida"
        dog_context = self._dog_context_text(request)
        urgency_rule = (
            "El mensaje contiene senales de urgencia. Indica acudir a un veterinario de urgencias ahora."
            if safety_context.get("urgent")
            else "Si detectas senales graves, recomienda veterinario."
        )
        system = (
            "Eres Aura Assistant, un asistente de orientacion basica para cuidado canino.\n"
            "Responde unicamente en espanol, breve y claro.\n"
            "No uses portugues ni ingles.\n"
            "Usa el contexto como referencia, pero no inventes edad, peso, diagnostico, raza ni sintomas.\n"
            "No recomiendes medicamentos, dosis ni tratamientos medicos.\n"
            "Si falta informacion especifica del perro, pide aclaracion concreta.\n"
            "Si el contexto menciona esperanza de vida, aclara que es informacion general de raza.\n"
            f"{urgency_rule}"
        )
        user = (
            f"Intencion: {intent}\n"
            f"Raza detectada: {breed_name}\n"
            f"Datos del perro: {dog_context}\n"
            f"Contexto local de Aura:\n{context}\n\n"
            f"Pregunta del usuario: {request.message}\n"
            "Respuesta:"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _ask_hugging_face(self, messages: list[dict[str, str]]) -> str:
        if not settings.HF_TOKEN:
            raise RuntimeError("HF_TOKEN is not configured")
        payload = {
            "model": settings.HF_MODEL,
            "messages": messages,
            "max_tokens": settings.HF_MAX_TOKENS,
            "temperature": settings.HF_TEMPERATURE,
        }
        response = requests.post(
            settings.HF_BASE_URL,
            headers={
                "Authorization": f"Bearer {settings.HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.HF_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Hugging Face returned HTTP {response.status_code}: {self._safe_body(response)}")

        data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected Hugging Face response format") from exc

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
            safety_level="hf_unavailable",
            disclaimer=DEFAULT_DISCLAIMER if request.include_disclaimer else None,
            recommend_vet=bool(safety_context.get("recommend_vet")),
            sources=["hf_rag", "aura_assistant_docs"],
            suggested_followups=[
                "Verifica que HF_TOKEN este configurado en el backend.",
                "Verifica que HF_MODEL este disponible en Hugging Face.",
                "Intenta de nuevo en unos segundos.",
            ],
        )

    def _retrieval_query(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
    ) -> str:
        breed_name = plain_text((breed_info or {}).get("name")) if breed_info else ""
        return " ".join(part for part in (request.message, intent, breed_name) if part)

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

    def _breed_out(self, breed_info: Optional[Dict[str, Any]]) -> AssistantBreedOut | None:
        if not breed_info:
            return None
        return AssistantBreedOut(
            label=breed_info.get("slug") or breed_info.get("label"),
            name=plain_text(breed_info.get("name")),
        )

    def _sources(self, retrieved) -> list[str]:
        sources = []
        for item in retrieved:
            if item.source and item.source not in sources:
                sources.append(item.source)
        return sources

    def _safe_body(self, response: requests.Response) -> str:
        body = self._redact_token(response.text.strip())
        if len(body) > 240:
            return body[:240] + "..."
        return body

    def _safe_error(self, exc: Exception) -> str:
        return self._redact_token(str(exc))

    def _redact_token(self, text: str) -> str:
        if settings.HF_TOKEN:
            return text.replace(settings.HF_TOKEN, "[redacted]")
        return text
