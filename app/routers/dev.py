# app/routers/dev.py
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlmodel import Session
from app.db import create_db_and_tables, get_session
from app.core.config import settings

router = APIRouter(prefix="/_dev", tags=["dev"])

@router.post("/init-db")
def init_db():
    create_db_and_tables()
    return {"detail": "Tablas creadas/migradas."}

@router.get("/db-info")
def db_info(session: Session = Depends(get_session)):
    # SELECT version() devuelve un row; lo convertimos a str
    row = session.exec(text("select version()")).first()
    # row puede ser tuple/Row -> tomamos la primera columna
    version = row[0] if isinstance(row, (list, tuple)) else str(row)
    return {
        "db_url": str(settings.DB_URL),
        "version": version,
    }
