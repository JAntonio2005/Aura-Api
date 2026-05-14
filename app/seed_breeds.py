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


def load_breeds_json() -> List[Dict[str, Any]]:
    candidates = [
        Path("app/data/breeds.json"),
        Path("exports/static/breeds.json"),
    ]

    for path in candidates:
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "items" in raw:
            return raw["items"]
        if isinstance(raw, list):
            return raw

        raise ValueError("Formato de breeds.json no soportado (esperaba lista o dict con 'items').")

    raise FileNotFoundError("No se encontró breeds.json en app/data ni en exports/static.")


def normalize_breed_data(data: Dict[str, Any]) -> Dict[str, Any]:
    slug = data.get("slug")
    if not slug:
        raise ValueError("Cada raza debe incluir slug.")

    label = data.get("label")
    if not label:
        label = slug.split("-", 1)[1] if "-" in slug else slug

    name = data.get("name")
    if not name:
        name = label.replace("-", " ").replace("_", " ").title()

    normalized = dict(data)
    normalized["label"] = label
    normalized["name"] = name
    return normalized


def seed_breeds():
    breeds = load_breeds_json()

    # Asegura carpetas estáticas mínimas
    (STATIC_DIR / "breeds").mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        for item in breeds:
            upsert_breed(session, normalize_breed_data(item))
        session.commit()


if __name__ == "__main__":
    seed_breeds()
