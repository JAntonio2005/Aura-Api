from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


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


def plain_text(value: Any) -> str:
    text = str(value or "").strip()
    try:
        text = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        pass
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = plain_text(text)
    return re.sub(r"\s+", " ", text.strip().lower())


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
