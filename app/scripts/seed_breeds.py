import json
from pathlib import Path
from typing import Any
from sqlmodel import Session, select
from app.db import get_session, engine
from app.models.breed import Breed

BREEDS_JSON = Path("exports/static/breeds.json")

def load_json():
    candidates = [Path("app/data/breeds.json"), BREEDS_JSON]

    for path in candidates:
        if not path.exists():
            continue

        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "items" in raw:
            return raw["items"]
        if isinstance(raw, list):
            return raw
        raise ValueError("Formato de breeds.json no soportado (esperaba lista o dict con 'items').")

    raise FileNotFoundError("No se encontró breeds.json en app/data ni en exports/static.")


def normalize_breed_data(data: dict[str, Any]) -> dict[str, Any]:
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

def upsert_breed(session: Session, data: dict):
    # normaliza imágenes: asegúrate que sean rutas /static...
    images = data.get("images", [])[:6]
    # busca por label (clave del modelo)
    b = session.exec(select(Breed).where(Breed.label == data["label"])).first()
    if not b:
        b = Breed(
            label=data["label"],
            slug=data["slug"],
            name=data.get("name"),
            description=data.get("description"),
            temperament=data.get("temperament"),
            origin=data.get("origin"),
            size=data.get("size"),
            life_span=data.get("life_span"),
            images=images,
        )
        session.add(b)
    else:
        b.slug = data.get("slug", b.slug)
        b.name = data.get("name", b.name)
        b.description = data.get("description", b.description)
        b.temperament = data.get("temperament", b.temperament)
        b.origin = data.get("origin", b.origin)
        b.size = data.get("size", b.size)
        b.life_span = data.get("life_span", b.life_span)
        b.images = images
    session.commit()

def main():
    data = load_json()
    with Session(engine) as session:
        for item in data:
            upsert_breed(session, normalize_breed_data(item))
    print("Catálogo de razas importado/actualizado ✅")

if __name__ == "__main__":
    main()
