# app/core/security.py
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.core.config import settings
from uuid import uuid4
import re
import secrets

# ====== Hashing / Verificación ======
pwd_context = CryptContext(
    schemes=["bcrypt_sha256"],  # seguimos con bcrypt_sha256
    deprecated="auto",
)

def hash_password(p: str) -> str:
    return pwd_context.hash(p)

def verify_password(p: str, hashed: str) -> bool:
    return pwd_context.verify(p, hashed)

# ====== Política de contraseñas ======
def password_policy_ok(p: str) -> tuple[bool, str]:
    """
    Reglas mínimas:
    - >= 8 caracteres
    - al menos 1 mayúscula, 1 minúscula y 1 dígito
    (Con bcrypt_sha256 no necesitamos truncar a 72 bytes)
    """
    if len(p) < 8:
        return False, "La contraseña debe tener al menos 8 caracteres."
    if not re.search(r"[A-Z]", p):
        return False, "Debe incluir al menos una mayúscula."
    if not re.search(r"[a-z]", p):
        return False, "Debe incluir al menos una minúscula."
    if not re.search(r"\d", p):
        return False, "Debe incluir al menos un dígito."
    return True, ""

# ====== JWT de acceso ======
def create_access_token(data: dict, *, token_version: int = 0, expires_minutes: int | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    # jti único por token + versión para soportar logout-all
    to_encode.update({
        "exp": expire,
        "jti": str(uuid4()),
        "tv": token_version,  # token_version
    })
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])


def get_claim(token: str, key: str, default=None):
    try:
        payload = decode_token(token)
        return payload.get(key, default)
    except JWTError:
        return default
# ====== Tokens de reseteo (para "olvidé mi contraseña") ======
def new_reset_token() -> str:
    """Token URL-safe; en prod se envía por email."""
    return secrets.token_urlsafe(32)

def minutes_from_now(m: int):
    return datetime.utcnow().replace(tzinfo=None) + timedelta(minutes=m)
