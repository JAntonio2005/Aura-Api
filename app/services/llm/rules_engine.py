from __future__ import annotations

# Legacy engine. Not used by the default Aura Assistant RAG flow.

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from app.models.schemas import AssistantBreedOut, AssistantRequest, AssistantResponse
from app.services.llm.base import AssistantEngine


BREEDS_JSON_PATH = Path("app/data/breeds.json")
DEFAULT_DISCLAIMER = "Esta orientacion no reemplaza una consulta veterinaria."
VALID_INTENTS = {"care", "feeding", "health", "adoption", "rescue", "general"}

INTENT_KEYWORDS = {
    "feeding": ["aliment", "comida", "comer", "croqueta", "dieta", "nutricion"],
    "health": [
        "enfer",
        "salud",
        "vomit",
        "diarrea",
        "tos",
        "dolor",
        "herida",
        "sangrado",
        "convulsion",
        "respirar",
        "intoxic",
    ],
    "adoption": ["adopt", "adopcion", "hogar", "familia", "casa"],
    "rescue": ["rescat", "calle", "abandono", "refugio", "casa hogar", "temporal"],
    "care": ["cuidado", "cuidados", "bano", "paseo", "ejercicio", "higiene", "cepill"],
}

URGENT_PATTERNS = [
    "convulsion",
    "dificultad para respirar",
    "no puede respirar",
    "sangrado",
    "hemorragia",
    "intoxic",
    "veneno",
    "vomito persistente",
    "vomita mucho",
    "diarrea severa",
    "diarrea con sangre",
    "inconsciente",
    "inconsciencia",
    "desmayo",
]


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = plain_text(text)
    return re.sub(r"\s+", " ", text.strip().lower())


def plain_text(value: Any) -> str:
    text = str(value or "").strip()
    try:
        text = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        pass
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


@lru_cache(maxsize=1)
def load_breeds() -> list[Dict[str, Any]]:
    raw = json.loads(BREEDS_JSON_PATH.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "items" in raw:
        return raw["items"]
    if isinstance(raw, list):
        return raw
    return []


def infer_intent(message: str, requested_intent: Optional[str]) -> str:
    intent = normalize_text(requested_intent)
    if intent in VALID_INTENTS:
        return intent

    text = normalize_text(message)
    for candidate, keywords in INTENT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return candidate
    return "general"


def find_breed(value: Optional[str]) -> Optional[Dict[str, Any]]:
    needle = normalize_text(value)
    if not needle:
        return None

    for item in load_breeds():
        candidates = [
            item.get("label"),
            item.get("canonical_label"),
            item.get("slug"),
            item.get("name"),
            item.get("display_name"),
        ]
        if needle in {normalize_text(candidate) for candidate in candidates}:
            return item
    return None


def has_urgent_symptom(message: str, dog_context: Optional[Dict[str, Any]]) -> bool:
    context_text = ""
    if dog_context:
        context_text = " ".join(str(value) for value in dog_context.values())
    text = normalize_text(f"{message} {context_text}")
    return any(pattern in text for pattern in URGENT_PATTERNS)


def build_safety_context(
    message: str,
    dog_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    urgent = has_urgent_symptom(message, dog_context)
    return {
        "urgent": urgent,
        "safety_level": "urgent_vet" if urgent else "basic_guidance",
        "recommend_vet": urgent,
    }


def suggested_followups(intent: str, urgent: bool) -> list[str]:
    if urgent:
        return [
            "Hace cuanto empezaron los sintomas?",
            "El perro esta consciente y respirando con normalidad?",
            "Puedes contactar a un veterinario de urgencias ahora?",
        ]

    by_intent = {
        "feeding": [
            "Que edad aproximada tiene?",
            "Esta bajo peso o ha dejado de comer?",
            "Tiene vomito o diarrea despues de comer?",
        ],
        "health": [
            "Desde cuando notas los sintomas?",
            "Come, toma agua y respira normal?",
            "Hay fiebre, dolor o decaimiento?",
        ],
        "adoption": [
            "Vive con ninos u otros animales?",
            "La familia ya tuvo perros antes?",
            "Hay posibilidad de seguimiento posterior?",
        ],
        "rescue": [
            "El perro permite acercamiento?",
            "Tiene heridas visibles?",
            "Ya tienes transporte o casa temporal?",
        ],
        "care": [
            "Es cachorro, adulto o senior?",
            "Vive en interior o exterior?",
            "Tiene vacunas o desparasitacion al dia?",
        ],
        "general": [
            "Quieres orientacion de alimentacion, salud o adopcion?",
            "Conoces la raza o tamano del perro?",
            "Hay algun sintoma o condicion especial?",
        ],
    }
    return by_intent.get(intent, by_intent["general"])


def _breed_out(breed: Optional[Dict[str, Any]]) -> Optional[AssistantBreedOut]:
    if not breed:
        return None
    return AssistantBreedOut(
        label=breed.get("slug") or breed.get("label"),
        name=plain_text(breed.get("name")),
    )


def _sources(breed: Optional[Dict[str, Any]], extra: Optional[list[str]] = None) -> list[str]:
    sources = ["general_care_rules"]
    if breed:
        sources.insert(0, str(BREEDS_JSON_PATH).replace("\\", "/"))
    if extra:
        sources.extend(extra)
    return sources


def _breed_sentence(breed: Optional[Dict[str, Any]]) -> str:
    if not breed:
        return ""

    details = [f"Para {plain_text(breed.get('name') or 'esta raza')}"]
    if breed.get("size"):
        details.append(f"de tamano {plain_text(breed['size']).lower()}")
    if breed.get("temperament"):
        temperament = plain_text(breed["temperament"]).rstrip(".")
        details.append(f"con temperamento {temperament.lower()}")
    return " ".join(details) + ", "


def _context_sentence(dog_context: Optional[Dict[str, Any]]) -> str:
    if not dog_context:
        return ""

    parts = []
    age = dog_context.get("age")
    condition = dog_context.get("condition")
    if age:
        parts.append(f"edad: {age}")
    if condition:
        parts.append(f"condicion: {condition}")
    if not parts:
        return ""
    return "Considerando el contexto del perro (" + "; ".join(parts) + "), "


def _with_prefix(prefix: str, sentence: str) -> str:
    if not prefix:
        return sentence
    return prefix + sentence[:1].lower() + sentence[1:]


def build_rule_answer(
    intent: str,
    breed: Optional[Dict[str, Any]],
    dog_context: Optional[Dict[str, Any]],
    urgent: bool,
) -> str:
    if urgent:
        return (
            "Los sintomas que mencionas pueden indicar una emergencia. Manten al perro en un lugar "
            "seguro y tranquilo, evita darle medicamentos o dosis por tu cuenta y contacta a un "
            "veterinario o servicio de urgencias lo antes posible."
        )

    prefix = _breed_sentence(breed)
    context = _context_sentence(dog_context)

    if intent == "feeding":
        return (
            _with_prefix(prefix, "La alimentacion debe ser gradual, limpia y adecuada a su etapa de vida. ")
            + f"{context}ofrece agua fresca, porciones pequenas y alimento balanceado para perro. "
            "Si viene de rescate o bajo peso, evita sobrealimentarlo de golpe y consulta al veterinario "
            "para ajustar cantidades."
        )

    if intent == "health":
        return (
            _with_prefix(prefix, "Observa energia, apetito, hidratacion, evacuaciones y respiracion. ")
            + f"{context}si hay dolor intenso, fiebre, deshidratacion, heridas profundas o sintomas que "
            "no mejoran, lo correcto es acudir al veterinario. No recomiendo medicamentos ni dosis."
        )

    if intent == "adoption":
        return (
            _with_prefix(prefix, "Una adopcion responsable empieza con entrevista, revision del espacio, compromiso ")
            + "de esterilizacion cuando aplique y seguimiento posterior. Asegura una transicion tranquila, "
            "rutina estable y paciencia durante las primeras semanas."
        )

    if intent == "rescue":
        return (
            _with_prefix(prefix, "En un rescate prioriza seguridad: acercamiento lento, agua, sombra o abrigo, ")
            + "separacion de otros animales y registro con fotos. Si el perro esta herido, muy debil o "
            "asustado, pide apoyo de un veterinario o rescatista con experiencia."
        )

    if intent == "care":
        return (
            _with_prefix(prefix, "Los cuidados basicos incluyen agua limpia, alimento adecuado, descanso, higiene, ")
            + "paseos controlados, vacunas/desparasitacion al dia y observacion diaria de cambios de conducta. "
            f"{context}manten una rutina predecible y evita cambios bruscos."
        )

    return (
        _with_prefix(prefix, "Puedo orientarte sobre cuidados, alimentacion, salud preventiva, adopcion y rescate. ")
        + "Comparte edad aproximada, condicion, sintomas si existen y contexto de la situacion para darte "
        "una guia mas util y segura."
    )


class RulesEngine(AssistantEngine):
    name = "rules"

    def generate(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
        safety_context: Dict[str, Any],
    ) -> AssistantResponse:
        urgent = bool(safety_context.get("urgent"))
        return AssistantResponse(
            answer=build_rule_answer(intent, breed_info, request.dog_context, urgent),
            intent=intent,
            breed=_breed_out(breed_info),
            safety_level=safety_context.get("safety_level", "basic_guidance"),
            disclaimer=DEFAULT_DISCLAIMER if request.include_disclaimer else None,
            recommend_vet=bool(safety_context.get("recommend_vet")),
            sources=_sources(breed_info),
            suggested_followups=suggested_followups(intent, urgent),
        )
