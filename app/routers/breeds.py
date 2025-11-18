# app/routers/breeds.py
from __future__ import annotations

from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlmodel import Session, select
from sqlalchemy import func

from app.db import get_session
from app.models.breed import Breed
from app.models.history import SearchLog
from app.deps_optional import get_current_user_optional  # usuario opcional

router = APIRouter(prefix="/breeds", tags=["breeds"])


# ----------------------------
# Listado + búsqueda con paginación
# ----------------------------
@router.get("/")
def list_breeds(
    q: Optional[str] = Query(None, description="Buscar por nombre/label/slug (case-insensitive)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    stmt = select(Breed)
    if q:
        like = f"%{q.lower()}%"
        stmt = stmt.where(
            (func.lower(Breed.name).like(like))
            | (func.lower(Breed.label).like(like))
            | (func.lower(Breed.slug).like(like))
        )

    # total real (sin limit/offset)
    count_stmt = stmt.with_only_columns(func.count()).order_by(None)
    total = session.exec(count_stmt).one()

    items = session.exec(stmt.offset(offset).limit(limit)).all()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
    }


# ----------------------------
# Obtener ficha por SLUG
# ----------------------------
@router.get("/{slug}")
def get_breed_by_slug(
    slug: str,
    session: Session = Depends(get_session),
    user = Depends(get_current_user_optional),
):
    breed = session.exec(select(Breed).where(Breed.slug == slug)).first()
    if not breed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Raza no encontrada",
        )

    # registra consulta en historial si hay usuario
    if user:
        session.add(
            SearchLog(
                user_id=user.id,
                query_label=slug,             # 👈 aquí usamos el campo correcto
                created_at=datetime.utcnow(),
            )
        )
        session.commit()

    return {
        "slug": breed.slug,
        "label": breed.label,
        "name": breed.name,
        "description": breed.description,
        "temperament": breed.temperament,
        "origin": breed.origin,
        "size": breed.size,
        "life_span": breed.life_span,
        "images": breed.images,  # 6 URLs
    }


# ----------------------------
# Obtener ficha por LABEL (útil para /predict)
# ----------------------------
@router.get("/by-label/{label}")
def get_breed_by_label(
    label: str,
    session: Session = Depends(get_session),
    user = Depends(get_current_user_optional),
):
    breed = session.exec(select(Breed).where(Breed.label == label)).first()
    if not breed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Raza no encontrada para ese label",
        )

    # registra consulta en historial si hay usuario
    if user:
        session.add(
            SearchLog(
                user_id=user.id,
                query_label=label,            # 👈 igual aquí
                created_at=datetime.utcnow(),
            )
        )
        session.commit()

    return {
        "slug": breed.slug,
        "label": breed.label,
        "name": breed.name,
        "description": breed.description,
        "temperament": breed.temperament,
        "origin": breed.origin,
        "size": breed.size,
        "life_span": breed.life_span,
        "images": breed.images,
    }


# ----------------------------
# Solo labels (para autocompletar en el front)
# ----------------------------
@router.get("/labels")
def list_labels(
    q: Optional[str] = Query(None, description="Filtro por label/nombre/slug"),
    limit: int = Query(100, ge=1, le=500),
    session: Session = Depends(get_session),
):
    stmt = select(Breed.label, Breed.name, Breed.slug).order_by(Breed.label.asc())
    if q:
        like = f"%{q.lower()}%"
        stmt = stmt.where(
            (func.lower(Breed.label).like(like))
            | (func.lower(Breed.name).like(like))
            | (func.lower(Breed.slug).like(like))
        )

    rows = session.exec(stmt.limit(limit)).all()
    # normaliza a lista de objetos simple
    return [{"label": r[0], "name": r[1], "slug": r[2]} for r in rows]


# ----------------------------
# Sugerencias rápidas (alias simple)
# ----------------------------
@router.get("/suggest")
def suggest(
    q: str = Query(..., min_length=1, description="Texto para autocompletar"),
    limit: int = Query(10, ge=1, le=50),
    session: Session = Depends(get_session),
):
    like = f"%{q.lower()}%"
    stmt = (
        select(Breed.label, Breed.name, Breed.slug)
        .where(
            (func.lower(Breed.label).like(like))
            | (func.lower(Breed.name).like(like))
            | (func.lower(Breed.slug).like(like))
        )
        .order_by(Breed.label.asc())
        .limit(limit)
    )
    rows = session.exec(stmt).all()
    return [{"label": r[0], "name": r[1], "slug": r[2]} for r in rows]
