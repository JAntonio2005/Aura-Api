from pathlib import Path
from typing import Tuple, List
import tensorflow_datasets as tfds
from PIL import Image
from app.core.config import SAMPLES_DIR
from app.services.model import CLASS_NAMES, display_name

def _resolve_label(label_or_idx: str | int) -> Tuple[str, int]:
    if isinstance(label_or_idx, str) and label_or_idx.isdigit():
        idx = int(label_or_idx)
        return CLASS_NAMES[idx], idx
    if isinstance(label_or_idx, int):
        return CLASS_NAMES[label_or_idx], label_or_idx
    if label_or_idx in CLASS_NAMES:
        return label_or_idx, CLASS_NAMES.index(label_or_idx)
    # por nombre legible
    low = label_or_idx.lower()
    for i, s in enumerate(CLASS_NAMES):
        if display_name(s).lower() == low:
            return s, i
    raise ValueError(f"Label no encontrado: {label_or_idx}")

def ensure_samples(label_or_idx: str | int, k: int = 6) -> Tuple[List[str], str, int]:
    canon, idx = _resolve_label(label_or_idx)
    out_dir = SAMPLES_DIR / canon
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.jpg"))
    if len(existing) >= k:
        urls = [f"/static/samples/{canon}/{p.name}" for p in existing[:k]]
        return urls, canon, idx

    ds = tfds.load("stanford_dogs", split="train+test", as_supervised=True, shuffle_files=False)
    saved = existing
    need = k - len(existing)

    for img, y in ds:
        if int(y.numpy()) == idx:
            pil = Image.fromarray(img.numpy()).convert("RGB").resize((256, 256))
            fname = f"{display_name(canon).replace(' ', '_').lower()}_{len(saved)+1:02d}.jpg"
            fpath = out_dir / fname
            pil.save(fpath, "JPEG", quality=85, optimize=True)
            saved.append(fpath)
            if len(saved) >= k: break

    urls = [f"/static/samples/{canon}/{p.name}" for p in sorted(out_dir.glob('*.jpg'))[:k]]
    return urls, canon, idx
