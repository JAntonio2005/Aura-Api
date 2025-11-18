# app/main.py
from fastapi import FastAPI, APIRouter, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from sqlalchemy import text
from sqlmodel import Session

from app.core.config import settings, STATIC_DIR
from app.routers import predict, breeds, auth, history, account
import app.models.history          # registra PredictionLog / SearchLog
import app.models.password_reset   # registra PasswordReset
from app.db import create_db_and_tables, get_session
from app.routers import dev
app = FastAPI(title="Dog Breed Classifier", version="1.1")

print(f"[DB] URL activa: {settings.DB_URL}")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static (samples)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Routers
app.include_router(auth.router)      # /auth/*
app.include_router(predict.router)   # /predict (protegido)
app.include_router(breeds.router)    # /labels, /breed_info, /samples
app.include_router(history.router)   # /history/*
app.include_router(account.router)   # /account/*
app.include_router(dev.router)    # /_dev/* (solo si DEBUG)

# Crear tablas al iniciar (una sola vez aquí)
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# ----- Rutas de desarrollo opcionales -----
if settings.DEBUG:
    dev = APIRouter(tags=["_dev"])

    @dev.post("/_dev/init-db")
    def init_db():
        create_db_and_tables()
        return {"detail": "Tablas creadas/migradas."}

    @dev.get("/_dev/db-info")
    def db_info(session: Session = Depends(get_session)):
        ver = session.exec(text("select version()")).first()
        return {"db_url": str(settings.DB_URL), "version": ver}

    app.include_router(dev)
