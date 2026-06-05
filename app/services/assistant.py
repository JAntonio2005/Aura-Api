from __future__ import annotations

from typing import Any, Dict, Optional

from app.models.schemas import AssistantRequest
from app.services.assistant_context import (
    build_safety_context,
    find_breed,
    find_breed_in_text,
    infer_intent,
)
from app.services.llm.factory import get_assistant_engine


def build_assistant_response(
    *,
    message: str,
    requested_intent: Optional[str],
    breed_value: Optional[str],
    dog_context: Optional[Dict[str, Any]],
    language: str = "es",
    include_disclaimer: bool = True,
) -> Dict[str, Any]:
    request = AssistantRequest(
        message=message,
        intent=requested_intent,
        breed=breed_value,
        dog_context=dog_context,
        language=language,
        include_disclaimer=include_disclaimer,
    )
    intent = infer_intent(request.message, request.intent)
    breed_info = find_breed(request.breed) or find_breed_in_text(request.message)
    safety_context = build_safety_context(request.message, request.dog_context)
    engine = get_assistant_engine()
    response = engine.generate(request, breed_info, intent, safety_context)
    return response.model_dump()
