# app/core/config.py
from __future__ import annotations

from pathlib import Path
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import EmailStr

# ----------------------------
# Rutas base del proyecto
# AURA-ML2/
# ├─ exports/
# └─ app/
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORTS_DIR  = PROJECT_ROOT / "exports"

def resolve_model_dir(base: Path = EXPORTS_DIR) -> Path:
    """
    Devuelve la exportación más reciente que contenga model.keras y labels.json.
    Lanza FileNotFoundError si no encuentra ninguna.
    """
    candidates = sorted(base.glob("stanford-dogs_*"))
    for d in reversed(candidates):
        if (d / "model.keras").exists() and (d / "labels.json").exists():
            return d
    raise FileNotFoundError(
        "No se encontró export con model.keras y labels.json en /exports"
    )

# Permite forzar el directorio del modelo vía env MODEL_DIR; si no, autodetecta.
MODEL_DIR_ENV = os.getenv("MODEL_DIR")
if MODEL_DIR_ENV:
    MODEL_DIR = Path(MODEL_DIR_ENV)
else:
    try:
        MODEL_DIR = resolve_model_dir()
    except FileNotFoundError:
        # No hay modelo aún; algunas rutas lo manejan en runtime.
        MODEL_DIR = EXPORTS_DIR

IMG_SIZE = 224

# ----------------------------
# Static (para /static y /static/samples)
# ----------------------------
STATIC_DIR  = (EXPORTS_DIR / "static").resolve()
SAMPLES_DIR = STATIC_DIR / "samples"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Settings (JWT / DB / Email)
# ----------------------------
class Settings(BaseSettings):
    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change_me_super_secret")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    # DB (por defecto SQLite en el root del proyecto)
    DB_URL: str = os.getenv("DB_URL", f"sqlite:///{(PROJECT_ROOT / 'data.db').as_posix()}")

    # Frontend: URL que recibirá ?token= para reset password
    FRONTEND_RESET_URL: str = os.getenv("FRONTEND_RESET_URL", "http://127.0.0.1:3000/reset-password")

    # Email (canónicos) — aceptan alias desde .env
    SMTP_HOST: Optional[str] = os.getenv("SMTP_HOST") or None
    SMTP_PORT: Optional[int] = int(os.getenv("SMTP_PORT", "0")) or None

    # Soportar ambos nombres de variable: SMTP_USERNAME/SMTP_USER
    SMTP_USERNAME: Optional[str] = (
        os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER") or None
    )
    # Soportar ambos nombres de variable: SMTP_PASSWORD/SMTP_PASS
    SMTP_PASSWORD: Optional[str] = (
        os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS") or None
    )
    # Soportar ambos nombres de variable: EMAIL_FROM/SMTP_FROM
    EMAIL_FROM: Optional[EmailStr] = (
        os.getenv("EMAIL_FROM") or os.getenv("SMTP_FROM") or None
    )

    # Modo
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # Carga automática de .env (pydantic-settings v2)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
