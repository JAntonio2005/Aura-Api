from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.services.assistant_context import normalize_text, plain_text
from app.services.rag.chunker import chunk_text, iter_supported_documents


DOCS_DIR = PROJECT_ROOT / "data" / "assistant_docs"
BREEDS_JSON = PROJECT_ROOT / "app" / "data" / "breeds.json"

STOPWORDS = {
    "a",
    "al",
    "como",
    "con",
    "cual",
    "cuando",
    "de",
    "del",
    "el",
    "en",
    "es",
    "esta",
    "este",
    "la",
    "las",
    "le",
    "lo",
    "los",
    "me",
    "mi",
    "para",
    "perro",
    "perros",
    "por",
    "que",
    "se",
    "si",
    "su",
    "sus",
    "tiene",
    "un",
    "una",
    "y",
}


@dataclass(frozen=True)
class SimpleRetrievedContext:
    text: str
    source: str
    score: int
    metadata: dict | None = None


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False)
    return path.read_text(encoding="utf-8")


def _tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]{3,}", normalized)
    return [token for token in tokens if token not in STOPWORDS]


def _source_for(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


@lru_cache(maxsize=1)
def _load_chunks() -> tuple[SimpleRetrievedContext, ...]:
    chunks: list[SimpleRetrievedContext] = []
    for path in iter_supported_documents(DOCS_DIR) or []:
        for index, chunk in enumerate(chunk_text(_read_document(path), chunk_size=900, overlap=120)):
            chunks.append(
                SimpleRetrievedContext(
                    text=plain_text(chunk),
                    source=_source_for(path),
                    score=0,
                    metadata={"chunk": index, "kind": "assistant_doc"},
                )
            )

    if BREEDS_JSON.exists():
        breed_data = json.loads(BREEDS_JSON.read_text(encoding="utf-8"))
        for index, chunk in enumerate(
            chunk_text(json.dumps(breed_data, ensure_ascii=False), chunk_size=900, overlap=120)
        ):
            chunks.append(
                SimpleRetrievedContext(
                    text=plain_text(chunk),
                    source=_source_for(BREEDS_JSON),
                    score=0,
                    metadata={"chunk": index, "kind": "breeds_json"},
                )
            )

    return tuple(chunks)


class SimpleRetriever:
    def __init__(self, *, top_k: int = 4):
        self.top_k = top_k

    def is_available(self) -> tuple[bool, str]:
        chunks = _load_chunks()
        if not chunks:
            return False, f"No assistant documents found in {DOCS_DIR}"
        return True, "ready"

    def retrieve_with_metadata(self, query: str, *, top_k: int | None = None) -> list[SimpleRetrievedContext]:
        chunks = _load_chunks()
        if not chunks:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return list(chunks[: top_k or self.top_k])

        scored: list[SimpleRetrievedContext] = []
        token_set = set(tokens)
        for item in chunks:
            haystack = normalize_text(f"{item.source} {item.text}")
            score = 0
            for token in token_set:
                occurrences = haystack.count(token)
                if occurrences:
                    score += occurrences
                    if token in normalize_text(item.source):
                        score += 2
            if score > 0:
                scored.append(
                    SimpleRetrievedContext(
                        text=item.text,
                        source=item.source,
                        score=score,
                        metadata=item.metadata,
                    )
                )

        if not scored:
            return list(chunks[: top_k or self.top_k])

        scored.sort(key=lambda item: item.score, reverse=True)
        limit = top_k or self.top_k
        selected: list[SimpleRetrievedContext] = []
        seen_sources: set[str] = set()
        for item in scored:
            if item.source in seen_sources:
                continue
            selected.append(item)
            seen_sources.add(item.source)
            if len(selected) >= limit:
                return selected

        for item in scored:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[str]:
        return [item.text for item in self.retrieve_with_metadata(query, top_k=top_k)]
