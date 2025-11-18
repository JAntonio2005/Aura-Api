from typing import List, Dict
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from app.core.config import MODEL_DIR, IMG_SIZE

MODEL_PATH = MODEL_DIR / "model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

model = tf.keras.models.load_model(str(MODEL_PATH))
CLASS_NAMES: List[str] = json.load(open(LABELS_PATH, "r", encoding="utf-8"))

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype("float32")
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, 0)

def predict_topk(img: Image.Image, k: int = 5) -> Dict:
    x = preprocess_pil(img)
    p = model.predict(x)
    probs = p[0]
    idx = int(np.argmax(probs))
    order = np.argsort(-probs)[:k]
    top5 = [{"index": int(i), "label": CLASS_NAMES[int(i)], "score": float(probs[i])} for i in order]
    return {"top1": top5[0], "top5": top5, "model_dir": str(MODEL_DIR)}

def display_name(label: str) -> str:
    # "n02085620-chihuahua" -> "Chihuahua"
    try:
        return label.split("-", 1)[1].replace("_", " ").title()
    except Exception:
        return label
