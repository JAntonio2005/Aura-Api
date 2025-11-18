# app/scripts/generate_breeds_json.py
import json
from pathlib import Path

# Ajusta este path si lo tienes diferente
BREEDS_ROOT = Path("exports/static/breeds")
OUTPUT_PATH = Path("exports/breeds.json")

def main():
    if not BREEDS_ROOT.exists():
        raise SystemExit(f"No encuentro la carpeta: {BREEDS_ROOT.resolve()}")

    items = []

    # Recorremos todas las carpetas de razas
    for folder in sorted(BREEDS_ROOT.iterdir()):
        if not folder.is_dir():
            continue

        # Ej: "n02085620-chihuahua"
        slug = folder.name.lower()

        # Parte después del id -> "chihuahua"
        try:
            label = slug.split("-", 1)[1]
        except IndexError:
            # Si alguna carpeta no sigue el formato, la brincamos
            print(f"Omitiendo carpeta con formato raro: {slug}")
            continue

        # "chihuahua" -> "Chihuahua"
        # "black-and-tan-coonhound" -> "Black And Tan Coonhound"
        name = label.replace("-", " ").title()

        # Rutas a las 6 imágenes
        images = [
            f"/static/breeds/{slug}/{i}.jpg"
            for i in range(1, 7)
        ]

        items.append({
            "slug": slug,
            "label": label,
            "name": name,
            "description": "",
            "temperament": "",
            "origin": "",
            "size": "",
            "life_span": "",
            "images": images
        })

    data = {
        "total": len(items),
        "items": items
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ breeds.json generado en: {OUTPUT_PATH.resolve()}")
    print(f"   Razas totales: {len(items)}")

if __name__ == "__main__":
    main()
