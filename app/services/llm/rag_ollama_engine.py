from __future__ import annotations

import json
import re
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
        context_update = self._context_update_response(request, breed_info, intent, safety_context)
        if context_update:
            return context_update

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

        answer = self._clean_answer(answer)
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
            suggested_followups=self._suggested_followups(intent, bool(safety_context.get("urgent"))),
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
        message_facts = self._message_facts_text(request, breed_info)
        clarification_rule = self._clarification_rule(request)
        return (
            "Eres Aura Assistant, un asistente de orientacion basica para cuidado canino.\n"
            "Reglas:\n"
            "- Responde unicamente en espanol, breve y claro.\n"
            "- No uses portugues ni ingles. Usa 'o' en lugar de 'ou'.\n"
            "- Usa el contexto recuperado solo como informacion general.\n"
            "- No inventes edad, peso, raza, diagnostico, sintomas, tiempo enfermo ni condicion fisica.\n"
            "- Si faltan datos especificos del perro, pide aclaracion en lugar de estimar.\n"
            "- Si el mensaje del usuario aporta un dato como edad o raza, reconocelo como dato del perro.\n"
            "- Si el mensaje solo aporta un dato conversacional, reconocelo y pide el siguiente dato util.\n"
            "- No reemplaces datos del usuario con razas o cifras del contexto recuperado.\n"
            "- Si el contexto menciona esperanza de vida de una raza, aclara que es expectativa general, no edad del perro.\n"
            "- Las preguntas de seguimiento deben servir para que el usuario aporte mas contexto.\n"
            "- No recomiendes medicamentos ni dosis.\n"
            "- Sugiere veterinario ante sintomas graves.\n"
            f"{clarification_rule}\n\n"
            f"Intencion: {intent}\n"
            f"Raza: {breed_name}\n"
            f"Datos especificos del perro proporcionados: {dog_context}\n"
            f"Datos detectados en el mensaje actual: {message_facts}\n"
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

    def _clean_answer(self, answer: str) -> str:
        text = plain_text(answer)
        replacements = {
            "Se apresentar sintomas graves": "Si presenta sintomas graves",
            "se apresentar sintomas graves": "si presenta sintomas graves",
            "consulte um veterinario": "consulta a un veterinario",
            "consulte un veterinario": "consulta a un veterinario",
            "procure um veterinario": "contacta a un veterinario",
            " ou ": " o ",
            " Ou ": " O ",
            "o cao": "el perro",
            "do cao": "del perro",
            "em caso de": "en caso de",
            "nao": "no",
            "voce": "tu",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text

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

    def _message_facts_text(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
    ) -> str:
        facts = []
        text = normalize_text(request.message)
        age_match = re.search(r"\b(\d{1,2})\s*(anos?|mes(?:es)?)\b", text)
        if age_match:
            unit = "anos" if age_match.group(2).startswith("ano") else "meses"
            facts.append(f"edad aproximada={age_match.group(1)} {unit}")
        if breed_info:
            facts.append(f"raza={plain_text(breed_info.get('name'))}")
        if request.dog_context:
            for key, value in request.dog_context.items():
                if value not in (None, ""):
                    facts.append(f"{plain_text(key)}={plain_text(value)}")
        return "; ".join(facts) if facts else "ninguno"

    def _context_update_response(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
        safety_context: Dict[str, Any],
    ) -> Optional[AssistantResponse]:
        text = normalize_text(request.message)
        if "?" in request.message or text.startswith(("que ", "como ", "cuando ", "donde ", "por que ")):
            return None

        if text in {"hola", "buenos dias", "buenas tardes", "buenas noches"}:
            return AssistantResponse(
                answer=(
                    "Hola, soy Aura Assistant. Puedo orientarte sobre cuidado, alimentacion, "
                    "rescate, adopcion o senales de urgencia en perros. Cuentame que necesitas."
                ),
                intent=intent,
                breed=self._breed_out(breed_info),
                safety_level=safety_context.get("safety_level", "basic_guidance"),
                disclaimer=DEFAULT_DISCLAIMER if request.include_disclaimer else None,
                recommend_vet=bool(safety_context.get("recommend_vet")),
                sources=["rag_ollama", "conversation_context"],
                suggested_followups=self._suggested_followups(intent, bool(safety_context.get("urgent"))),
            )

        age_match = re.search(r"\b(\d{1,2})\s*(anos?|mes(?:es)?)\b", text)
        is_short = len(text.split()) <= 5
        if age_match and is_short:
            unit = "anos" if age_match.group(2).startswith("ano") else "meses"
            age_text = f"{age_match.group(1)} {unit}"
            return AssistantResponse(
                answer=(
                    f"Entendido, tiene {age_text}. Para orientarte mejor, dime tambien "
                    "su raza o tamano y si tiene algun sintoma o condicion especial."
                ),
                intent=intent,
                breed=self._breed_out(breed_info),
                safety_level=safety_context.get("safety_level", "basic_guidance"),
                disclaimer=DEFAULT_DISCLAIMER if request.include_disclaimer else None,
                recommend_vet=bool(safety_context.get("recommend_vet")),
                sources=["rag_ollama", "conversation_context"],
                suggested_followups=self._suggested_followups(intent, bool(safety_context.get("urgent"))),
            )

        if breed_info and is_short and re.search(r"\b(es|raza|es un|es una)\b", text):
            breed_name = plain_text(breed_info.get("name"))
            return AssistantResponse(
                answer=(
                    f"Entendido, es un {breed_name}. Para darte una guia mas util, "
                    "dime su edad aproximada y si fue rescatado, esta enfermo o necesita cuidados generales."
                ),
                intent=intent,
                breed=self._breed_out(breed_info),
                safety_level=safety_context.get("safety_level", "basic_guidance"),
                disclaimer=DEFAULT_DISCLAIMER if request.include_disclaimer else None,
                recommend_vet=bool(safety_context.get("recommend_vet")),
                sources=["rag_ollama", "conversation_context"],
                suggested_followups=self._suggested_followups(intent, bool(safety_context.get("urgent"))),
            )

        return None

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

    def _suggested_followups(self, intent: str, urgent: bool) -> list[str]:
        if urgent:
            return [
                "¿Desde cuándo presenta ese problema?",
                "¿Está consciente y respirando con normalidad?",
                "¿Puedes contactar a un veterinario de urgencias ahora?",
            ]

        by_intent = {
            "feeding": [
                "¿Qué edad aproximada tiene?",
                "¿Ha dejado de comer, vomitado o tenido diarrea?",
                "¿Conoces su raza o tamaño?",
            ],
            "health": [
                "¿Desde cuándo presenta ese problema?",
                "¿Está consciente y respirando con normalidad?",
                "¿Ha dejado de comer, vomitado o tenido diarrea?",
            ],
            "adoption": [
                "¿Qué edad aproximada tiene?",
                "¿Convive con niños u otros animales?",
                "¿La familia puede darle tiempo, paseos y atención veterinaria?",
            ],
            "rescue": [
                "¿El perro permite que te acerques?",
                "¿Tiene heridas visibles o parece muy débil?",
                "¿Conoces su raza o tamaño?",
            ],
            "care": [
                "¿Qué edad aproximada tiene?",
                "¿Conoces su raza o tamaño?",
                "¿Ha dejado de comer, vomitado o tenido diarrea?",
            ],
            "general": [
                "¿Qué edad aproximada tiene?",
                "¿Conoces su raza o tamaño?",
                "¿Hay algún síntoma o condición especial?",
            ],
        }
        return by_intent.get(intent, by_intent["general"])
