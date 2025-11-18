import json
from pathlib import Path
from sqlmodel import Session, select
from app.db import get_session, engine
from app.models.breed import Breed

BREEDS_JSON = Path("exports/static/breeds.json")

def load_json():
    return json.loads(BREEDS_JSON.read_text(encoding="utf-8"))

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
            upsert_breed(session, item)
    print("Catálogo de razas importado/actualizado ✅")

if __name__ == "__main__":
    main()
