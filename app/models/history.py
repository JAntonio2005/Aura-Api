from typing import Optional, Dict, Any
from datetime import datetime

from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON


class PredictionLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")

    # nombre original del archivo subido
    image_name: Optional[str] = Field(default=None, max_length=255)

    # mejor predicción del modelo
    top1_label: str = Field(index=True, max_length=100)
    top1_score: float

    # top-5 completo (lista de dicts: {index, label, score})
    top5: Dict[str, Any] = Field(sa_column=Column(JSON))

    # fecha/hora en que se hizo la predicción
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class SearchLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    # usuario que hizo la búsqueda
    user_id: int = Field(index=True, foreign_key="user.id")

    # qué se buscó (puede ser slug, label, etc.)
    query_label: str = Field(index=True, max_length=100)

    # cuándo se hizo la búsqueda
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
