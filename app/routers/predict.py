from fastapi import APIRouter, UploadFile, File, Depends, Query
from sqlmodel import Session
from PIL import Image

from app.db import get_session
from app.deps import get_current_user         # este sí es obligatorio
from app.models.history import PredictionLog
from app.services.model import predict_topk, display_name
from app.services.samples import ensure_samples

router = APIRouter(tags=["predict"])

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

    # Guardar log
    log = PredictionLog(
        user_id=user.id,
        image_name=file.filename,
        top1_label=top1["label"],
        top1_score=float(top1["score"]),
        top5=result["top5"],
    )
    session.add(log)
    session.commit()
    session.refresh(log)

    if enrich:
        canon = top1["label"]
        urls, canon, idx = ensure_samples(canon, 6)
        result["breed_detail"] = {
            "info": {
                "display_name": display_name(canon),
                "canonical_label": canon,
                "index": idx,
                "traits": ["Inteligente", "Alerta", "Leal"],
                "care": {"exercise": "Bajo–moderado", "grooming": "Bajo", "training": "Consistente"},
            },
            "images": urls,
        }

    return {"log_id": log.id, **result}
