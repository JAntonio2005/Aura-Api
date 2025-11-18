import json, os
from pathlib import Path
from app.core.config import STATIC_DIR, MODEL_DIR

BREEDS_DIR = STATIC_DIR / "breeds"
BREEDS_DIR.mkdir(parents=True, exist_ok=True)

labels_path = MODEL_DIR / "labels.json"
labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))

def slugify(label: str) -> str:
    return label.lower().replace("_", "-").replace(" ", "-")

created = 0
for lbl in labels:
    slug = slugify(lbl)
    d = BREEDS_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    keep = d / ".keep"
    if not keep.exists():
        keep.write_text("", encoding="utf-8")
        created += 1

print(f"Listo. Carpetas creadas/norm.: {created}. Directorio: {BREEDS_DIR}")
