# app/models/breed.py
from __future__ import annotations
from typing import List, Optional
from sqlmodel import SQLModel, Field, Column, JSON
from datetime import datetime

class Breed(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(index=True, unique=True, nullable=False)  # ej. "chihuahua"
    label: str = Field(index=True, unique=True, nullable=False) # debe empatar con el label del modelo ML si aplica
    name: str = Field(nullable=False)                            # nombre legible: "Chihuahua"
    description: str = Field(nullable=False, default="")
    temperament: Optional[str] = Field(default=None)
    origin: Optional[str] = Field(default=None)
    size: Optional[str] = Field(default=None)                    # toy / small / medium / large
    life_span: Optional[str] = Field(default=None)               # ej. "12–18 years"
    images: List[str] = Field(sa_column=Column(JSON), default_factory=list)  # URLs (tamaño 6)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
