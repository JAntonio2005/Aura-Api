from fastapi import APIRouter

from app.models.schemas import AssistantRequest, AssistantResponse
from app.services.assistant import build_assistant_response


router = APIRouter(tags=["assistant"])


@router.post("/assistant", response_model=AssistantResponse)
def assistant(payload: AssistantRequest):
    return build_assistant_response(
        message=payload.message,
        requested_intent=payload.intent,
        breed_value=payload.breed,
        dog_context=payload.dog_context,
        language=payload.language,
        include_disclaimer=payload.include_disclaimer,
    )
