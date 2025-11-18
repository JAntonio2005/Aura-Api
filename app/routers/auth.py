from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select, delete

from app.db import get_session
from app.models.user import User
from app.models.schemas import UserCreate, UserOut, Token
from app.models.revoked_token import RevokedToken
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
)
from app.deps import get_current_user, oauth2_scheme

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201)
def register(payload: UserCreate, session: Session = Depends(get_session)):
    exists = session.exec(select(User).where(User.email == payload.email)).first()
    if exists:
        raise HTTPException(status_code=409, detail="Email ya registrado")

    user = User(
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        # token_version por defecto = 0 en el modelo
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return UserOut(id=user.id, email=user.email, full_name=user.full_name)


@router.post("/login", response_model=Token)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
):
    user = session.exec(select(User).where(User.email == form.username)).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales inválidas"
        )

    # Emitimos JWT con uid y la token_version actual del usuario
    access_token = create_access_token(
    {"sub": user.email, "uid": user.id},
    token_version=user.token_version
)
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, email=user.email, full_name=user.full_name)


@router.post("/logout", status_code=status.HTTP_200_OK)
def logout(
    token: str = Depends(oauth2_scheme),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Cierra la sesión actual: revoca SOLO este token añadiendo su jti a la blacklist.
    """
    payload = decode_token(token)
    jti = payload.get("jti")
    exp = payload.get("exp")

    if not jti or not exp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Token inválido."
        )

    # Limpiar revocados expirados
    session.exec(delete(RevokedToken).where(RevokedToken.expires_at < datetime.utcnow()))
    session.commit()
    # Registrar revocación del token actual (si no existe)
    exists = session.exec(select(RevokedToken).where(RevokedToken.jti == jti)).first()
    if not exists:
        session.add(
            RevokedToken(
                jti=jti,
                user_id=user.id,
                expires_at=datetime.utcfromtimestamp(exp),
            )
        )
    session.commit()
    return {"detail": "Sesión cerrada."}


@router.post("/logout-all", status_code=status.HTTP_200_OK)
def logout_all(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Cierra todas las sesiones: incrementa token_version en el usuario.
    Todos los JWT emitidos antes quedan inválidos.
    """
    user.token_version += 1
    session.add(user)
    session.commit()
    return {"detail": "Todas las sesiones cerradas."}
