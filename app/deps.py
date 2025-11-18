from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlmodel import Session, select
from app.core.security import decode_token
from app.db import get_session
from app.models.user import User
from typing import Optional
from app.models.revoked_token import RevokedToken



oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session)
) -> User:
    try:
        payload = decode_token(token)
        email: str | None = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Token inválido (sub).")

        # Chequeo de blacklist solo si el JWT trae jti
        jti = payload.get("jti")
        if jti:
            revoked = session.exec(
                select(RevokedToken).where(RevokedToken.jti == jti)
            ).first()
            if revoked:
                raise HTTPException(status_code=401, detail="Token revocado.")

        user = session.exec(select(User).where(User.email == email)).first()
        if not user:
            raise HTTPException(status_code=401, detail="Usuario no encontrado.")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado.")
def get_current_user_optional() -> Optional[User]:
    try:
        return _get_current_user()
    except Exception:
        return None
    
