from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlmodel import Session, select
from app.db import get_session
from app.deps import get_current_user
from app.models.history import PredictionLog, SearchLog
from app.models.schemas import PredictionLogOut, SearchLogOut

router = APIRouter(prefix="/history", tags=["history"])

@router.get("/predictions", response_model=list[PredictionLogOut])
def my_predictions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
    user = Depends(get_current_user),
):
    q = select(PredictionLog).where(PredictionLog.user_id == user.id)\
                             .order_by(PredictionLog.created_at.desc())\
                             .limit(limit).offset(offset)
    return session.exec(q).all()

@router.delete("/predictions/{log_id}", status_code=204)
def delete_prediction(
    log_id: int,
    session: Session = Depends(get_session),
    user = Depends(get_current_user),
):
    log = session.get(PredictionLog, log_id)
    if not log or log.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No encontrado")
    session.delete(log)
    session.commit()

@router.get("/searches", response_model=list[SearchLogOut])
def my_searches(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
    user = Depends(get_current_user),
):
    q = select(SearchLog).where(SearchLog.user_id == user.id)\
                         .order_by(SearchLog.created_at.desc())\
                         .limit(limit).offset(offset)
    return session.exec(q).all()
