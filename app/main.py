# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from sqlalchemy.engine import make_url

from app.core.config import settings, STATIC_DIR
from app.routers import predict, breeds, auth, history, account, assistant
import app.models.history          # registra PredictionLog / SearchLog
import app.models.password_reset   # registra PasswordReset
from app.db import create_db_and_tables
from app.routers import dev
app = FastAPI(title="Dog Breed Classifier", version="1.1")

def _safe_db_url(url: str) -> str:
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return url

print(f"[DB] URL activa: {_safe_db_url(settings.DB_URL)}")

# CORS
cors_origins = settings.CORS_ORIGINS or (["*"] if settings.DEBUG else [])
allow_credentials = settings.CORS_ALLOW_CREDENTIALS and "*" not in cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
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
app.include_router(assistant.router)  # /assistant
if settings.DEBUG:
    app.include_router(dev.router)    # /_dev/* (solo si DEBUG)

# Crear tablas al iniciar (una sola vez aquí)
@app.on_event("startup")
def on_startup():
    try:
        create_db_and_tables()
    except Exception as exc:
        if settings.DB_STARTUP_STRICT:
            raise
        print(f"[Startup] WARNING: no se pudo inicializar DB: {exc}")
