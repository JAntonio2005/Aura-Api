from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
from sqlmodel import Session, select

from app.db import engine
from app.core.config import STATIC_DIR
from app.models.breed import Breed


def default_images_for_slug(slug: str) -> List[str]:
    base = f"/static/breeds/{slug}"
    return [f"{base}/{i}.jpg" for i in range(1, 7)]


def upsert_breed(session: Session, data: Dict[str, Any]):
    slug = data["slug"]
    images = data.get("images") or default_images_for_slug(slug)
    if len(images) != 6:
        raise ValueError(f"{slug}: images debe tener 6 URLs, tiene {len(images)}")

    b = session.exec(select(Breed).where(Breed.slug == slug)).first()
    now = datetime.utcnow()

    if b:
        b.label       = data["label"]
        b.name        = data["name"]
        b.description = data.get("description", "")
        b.temperament = data.get("temperament")
        b.origin      = data.get("origin")
        b.size        = data.get("size")
        b.life_span   = data.get("life_span")
        b.images      = images
        b.updated_at  = now
        session.add(b)
    else:
        b = Breed(
            slug=slug,
            label=data["label"],
            name=data["name"],
            description=data.get("description", ""),
            temperament=data.get("temperament"),
            origin=data.get("origin"),
            size=data.get("size"),
            life_span=data.get("life_span"),
            images=images,
            created_at=now,
            updated_at=now,
        )
        session.add(b)


def seed_breeds():
    # JSON de semillas
    # OJO: ajusta esta ruta a donde REALMENTE está tu archivo
    seeds_path = Path("exports/static/breeds.json")  # por tu descripción, lo más probable es este path

    with open(seeds_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # aquí usamos el array correcto
    if isinstance(raw, dict) and "items" in raw:
        breeds = raw["items"]
    elif isinstance(raw, list):
        breeds = raw
    else:
        raise ValueError("Formato de breeds.json no soportado (esperaba lista o dict con 'items').")

    # Asegura carpetas estáticas mínimas
    (STATIC_DIR / "breeds").mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        for item in breeds:
            upsert_breed(session, item)
        session.commit()


if __name__ == "__main__":
    seed_breeds()
