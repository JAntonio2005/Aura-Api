from fastapi import APIRouter, UploadFile, File, Depends, Query
from sqlmodel import Session, select
from sqlalchemy import or_
from PIL import Image

from app.db import get_session
from app.deps import get_current_user         # este sí es obligatorio
from app.models.breed import Breed
from app.models.history import PredictionLog
from app.services.model import predict_topk, display_name, CONFIDENCE_THRESHOLD
from app.services.samples import ensure_samples

router = APIRouter(tags=["predict"])

DEFAULT_DESCRIPTION = "Descripcion no disponible para esta raza."
DEFAULT_BASIC_CARE = "Cuidados basicos no disponibles. Consulta con un veterinario para recomendaciones generales."
DEFAULT_FEEDING = "Alimentacion no disponible. Consulta con un veterinario para una dieta adecuada."

def _build_breed_info(
    canon_label: str,
    display: str,
    breed: Breed | None,
) -> dict:
    description = DEFAULT_DESCRIPTION
    if breed and breed.description:
        description = breed.description

    basic_care = DEFAULT_BASIC_CARE
    feeding = DEFAULT_FEEDING

    return {
        "display_name": display,
        "canonical_label": canon_label,
        "description": description,
        "basic_care": basic_care,
        "feeding": feeding,
    }

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    enrich: bool = Query(False),
    user = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    img = Image.open(file.file)
    result = predict_topk(img)  # {"top1": {...}, "top5": [...]}
    top1 = result["top1"]
    top1_score = float(top1["score"])
    recognized = top1_score >= CONFIDENCE_THRESHOLD

    # Guardar log
    log = PredictionLog(
        user_id=user.id,
        image_name=file.filename,
        top1_label=top1["label"],
        top1_score=top1_score,
        top5=result["top5"],
    )
    session.add(log)
    session.commit()
    session.refresh(log)

    if not recognized:
        return {
            "recognized": False,
            "message": "No se pudo reconocer un perro claramente. Intenta con una imagen más nítida.",
            "log_id": log.id,
            "top1": top1,
            "top5": result["top5"],
        }

    canon = top1["label"]
    display = display_name(canon)
    breed = session.exec(
        select(Breed).where(or_(Breed.slug == canon, Breed.label == canon))
    ).first()
    info = _build_breed_info(canon, display, breed)

    response = {
        "recognized": True,
        "breed": canon,
        "confidence": top1_score,
        "possible_breeds": result["top5"],
        "info": info,
        "log_id": log.id,
        "top1": top1,
        "top5": result["top5"],
        "model_dir": result.get("model_dir"),
    }

    if enrich:
        urls, canon, idx = ensure_samples(canon, 6)
        response["breed_detail"] = {
            "info": {
                **info,
                "index": idx,
                "traits": ["Inteligente", "Alerta", "Leal"],
                "care": {"exercise": "Bajo-moderado", "grooming": "Bajo", "training": "Consistente"},
            },
            "images": urls,
        }

    return response
