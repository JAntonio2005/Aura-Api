from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import PROJECT_ROOT
from app.models.schemas import AssistantBreedOut, AssistantRequest, AssistantResponse
from app.services.llm.base import AssistantEngine
from app.services.llm.rules_engine import (
    DEFAULT_DISCLAIMER,
    RulesEngine,
    _sources,
    plain_text,
    suggested_followups,
)


class PicoGptEngine(AssistantEngine):
    name = "picogpt"
    USEFUL_TERMS = {
        "agua",
        "alimento",
        "alimentacion",
        "comida",
        "veterinario",
        "higiene",
        "descanso",
        "rutina",
        "cuidado",
        "cuidados",
        "salud",
        "paseo",
        "vacunas",
        "desparasitacion",
        "seguro",
        "tranquilo",
        "observa",
    }
    PROMPT_ECHO_TERMS = {"pregunta:", "respuesta:", "raza:", "contexto:"}

    def __init__(
        self,
        fallback: Optional[AssistantEngine] = None,
        model_size: str = "124M",
        max_tokens: int = 25,
        timeout_seconds: int = 90,
    ):
        self.fallback = fallback or RulesEngine()
        self.model_size = model_size
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.picogpt_dir = self._find_picogpt_dir()
        self.models_dir = self.picogpt_dir / "models" if self.picogpt_dir else None

    def generate(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
        safety_context: Dict[str, Any],
    ) -> AssistantResponse:
        if safety_context.get("urgent"):
            print("[Assistant] PicoGPT bypassed: urgent safety context; using rules fallback.")
            return self.fallback.generate(request, breed_info, intent, safety_context)

        available, reason = self.is_available()
        if not available:
            print(f"[Assistant] PicoGPT unavailable: {reason}; using rules fallback.")
            return self.fallback.generate(request, breed_info, intent, safety_context)

        prompt = self._build_prompt(request, breed_info, intent)
        started = time.perf_counter()
        try:
            generated = self._run_picogpt(prompt)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"[Assistant] PicoGPT failed after {elapsed:.2f}s: {exc}; using rules fallback.")
            return self.fallback.generate(request, breed_info, intent, safety_context)

        answer = self._clean_output(generated)
        is_valid, rejection_reason = self._validate_quality(answer)
        if not is_valid:
            elapsed = time.perf_counter() - started
            print(
                "[Assistant] PicoGPT response rejected by quality gate "
                f"after {elapsed:.2f}s: {rejection_reason}; fallback to RulesEngine."
            )
            return self.fallback.generate(request, breed_info, intent, safety_context)

        elapsed = time.perf_counter() - started
        print(f"[Assistant] PicoGPT generated response in {elapsed:.2f}s and passed quality gate.")

        breed_out = None
        if breed_info:
            breed_out = AssistantBreedOut(
                label=breed_info.get("slug") or breed_info.get("label"),
                name=plain_text(breed_info.get("name")),
            )

        return AssistantResponse(
            answer=answer,
            intent=intent,
            breed=breed_out,
            safety_level=safety_context.get("safety_level", "basic_guidance"),
            disclaimer=DEFAULT_DISCLAIMER if request.include_disclaimer else None,
            recommend_vet=bool(safety_context.get("recommend_vet")),
            sources=_sources(breed_info, ["picogpt_experimental"]),
            suggested_followups=suggested_followups(intent, False),
        )

    def is_available(self) -> tuple[bool, str]:
        if not self.picogpt_dir:
            return False, "picogpt source directory not found"

        problems = []
        missing_modules = [
            module
            for module in ("regex", "requests", "tqdm", "tensorflow")
            if importlib.util.find_spec(module) is None
        ]
        if missing_modules:
            problems.append("missing Python modules: " + ", ".join(missing_modules))

        required_source = ["encoder.py", "gpt2.py", "utils.py"]
        missing_source = [
            filename for filename in required_source if not (self.picogpt_dir / filename).exists()
        ]
        if missing_source:
            problems.append("missing source files: " + ", ".join(missing_source))

        model_dir = self.models_dir / self.model_size
        required_weights = [
            "checkpoint",
            "encoder.json",
            "hparams.json",
            "model.ckpt.data-00000-of-00001",
            "model.ckpt.index",
            "model.ckpt.meta",
            "vocab.bpe",
        ]
        missing_weights = [filename for filename in required_weights if not (model_dir / filename).exists()]
        if missing_weights:
            problems.append(
                f"missing GPT-2 {self.model_size} files in {model_dir}: "
                + ", ".join(missing_weights)
            )

        if problems:
            return False, "; ".join(problems)

        return True, "ready"

    def _find_picogpt_dir(self) -> Optional[Path]:
        candidates = [
            PROJECT_ROOT / "picogpt" / "picogpt",
            PROJECT_ROOT / "picoGPT" / "picogpt",
            PROJECT_ROOT / "reference" / "picoGPT",
            PROJECT_ROOT / "reference" / "picoGPT" / "picogpt",
        ]
        for candidate in candidates:
            if (candidate / "gpt2.py").exists() and (candidate / "encoder.py").exists():
                return candidate
        return None

    def _build_prompt(
        self,
        request: AssistantRequest,
        breed_info: Optional[Dict[str, Any]],
        intent: str,
    ) -> str:
        breed_name = "No especificada"
        if breed_info:
            breed_name = plain_text(breed_info.get("name")) or "No especificada"

        dog_context = ""
        if request.dog_context:
            dog_context = f" Contexto: {json.dumps(request.dog_context, ensure_ascii=True)}."
        size_context = ""
        if breed_info and breed_info.get("size"):
            size_context = f" Tamano: {plain_text(breed_info.get('size'))}."
        return (
            f"Intent: {intent}. Raza: {breed_name}.{size_context}{dog_context}\n"
            f"Pregunta: {plain_text(request.message)}\n"
            "Respuesta breve en espanol sobre cuidado canino seguro:"
        )

    def _run_picogpt(self, prompt: str) -> str:
        script = f"""
import contextlib
import io
import json
import sys

sys.path.insert(0, {str(self.picogpt_dir)!r})
from gpt2 import main

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    out = main(
        {prompt!r},
        n_tokens_to_generate={self.max_tokens},
        model_size={self.model_size!r},
        models_dir={str(self.models_dir)!r},
    )
print(json.dumps({{"output": out}}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            cwd=str(self.picogpt_dir),
            check=True,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        return str(payload.get("output", ""))

    def _clean_output(self, text: str) -> str:
        answer = plain_text(text)
        answer = " ".join(answer.split())
        for marker in ("Respuesta breve en espanol sobre cuidado canino seguro:", "Respuesta:"):
            if marker in answer:
                answer = answer.split(marker, 1)[-1].strip()
        if len(answer) > 700:
            answer = answer[:700].rsplit(" ", 1)[0].strip()
        return answer

    def _validate_quality(self, answer: str) -> tuple[bool, str]:
        if not answer:
            return False, "empty response"
        if len(answer) < 20:
            return False, "response shorter than 20 characters"

        lowered = answer.lower()
        if any(term in lowered for term in self.PROMPT_ECHO_TERMS):
            return False, "response echoes prompt markers"

        words = re.findall(r"[a-zA-Z]+", lowered)
        if len(words) < 4:
            return False, "too few words"

        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        most_common_count = max(counts.values())
        if most_common_count >= 4 or most_common_count / max(len(words), 1) > 0.45:
            return False, "repetitive output"

        if not any(term in lowered for term in self.USEFUL_TERMS):
            return False, "no useful care vocabulary"

        return True, "ok"
