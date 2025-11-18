from typing import Optional
from fastapi import Header, Depends
from sqlmodel import Session, select
from jose import JWTError, jwt

from app.core.config import settings
from app.db import get_session
from app.models.user import User

ALGO = "HS256"

def _get_user_by_email(session: Session, email: str) -> Optional[User]:
    return session.exec(select(User).where(User.email == email)).first()

def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    session: Session = Depends(get_session),
) -> Optional[User]:
    # Sin header -> anónimo
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGO])
        email = payload.get("sub")
        if not email:
            return None
        user = _get_user_by_email(session, email)
        return user
    except JWTError:
        return None
