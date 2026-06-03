from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.models.schemas import AssistantRequest, AssistantResponse


class AssistantEngine(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
        safety_context: Dict[str, Any],
    ) -> AssistantResponse:
        raise NotImplementedError
