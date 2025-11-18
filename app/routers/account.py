# app/routers/account.py
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlmodel import Session, select

from app.db import get_session
from app.deps import get_current_user
from app.models.user import User
from app.models.password_reset import PasswordReset
from app.models.schemas import ChangePasswordIn, ForgotPasswordIn, ResetPasswordIn
from app.core.security import (
    verify_password, hash_password, password_policy_ok,
    new_reset_token, minutes_from_now
)
from app.core.config import settings
from app.core.email import send_email

router = APIRouter(prefix="/account", tags=["account"])


@router.post("/change-password")
def change_password(
    payload: ChangePasswordIn,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    # 1) verifica actual
    if not verify_password(payload.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Contraseña actual incorrecta."
        )

    # 2) política
    ok, msg = password_policy_ok(payload.new_password)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )

    # 3) no permitir igual a la actual
    if verify_password(payload.new_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La nueva contraseña no puede ser igual a la actual."
        )

    # 4) guardar
    user.hashed_password = hash_password(payload.new_password)
    session.add(user)
    session.commit()
    return {"detail": "Contraseña actualizada."}


@router.post("/forgot-password")
def forgot_password(
    payload: ForgotPasswordIn,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
):
    # Siempre responde 200 (no revelar si el email existe)
    user = session.exec(select(User).where(User.email == payload.email)).first()
    if user:
        # (Opcional) invalidar tokens pendientes no usados
        # En SQLite podrías simplemente marcarlos como usados o borrarlos;
        # aquí los dejamos y solo creamos uno nuevo válido.

        # Crea token y registra
        token = new_reset_token()
        pr = PasswordReset(
            user_id=user.id,
            token=token,
            expires_at=minutes_from_now(30),  # expira en 30 min
        )
        session.add(pr)
        session.commit()

        # Enlace de reset hacia tu front (defínelo en .env FRONTEND_RESET_URL)
        reset_link = f"{settings.FRONTEND_RESET_URL}?token={token}"

        # Cuerpo del correo (HTML + texto)
        try:
            with open("app/templates/reset_password.html", "r", encoding="utf-8") as f:
                html = f.read().replace("{{RESET_LINK}}", reset_link)
        except FileNotFoundError:
            html = f"""
            <p>Solicitaste restablecer tu contraseña.</p>
            <p>Haz clic aquí: <a href="{reset_link}">{reset_link}</a></p>
            <p>Este enlace expira en 30 minutos.</p>
            """
        text = f"Restablece tu contraseña con este enlace (expira en 30 min): {reset_link}"

        # Enviar en background
        background.add_task(
            send_email,
            to=user.email,
            subject="Restablecer contraseña",
            html=html,
            text_fallback=text,
        )

        # En DEV: devuelve token para probar rápido
        if settings.DEBUG:
            return {
                "detail": "Si el email existe, se envió un enlace de recuperación.",
                "testing_token": token,
            }

    return {"detail": "Si el email existe, se envió un enlace de recuperación."}


@router.post("/reset-password")
def reset_password(
    payload: ResetPasswordIn,
    session: Session = Depends(get_session),
):
    pr = session.exec(
        select(PasswordReset).where(PasswordReset.token == payload.token)
    ).first()

    if not pr:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token inválido."
        )
    if pr.used_at is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token ya utilizado."
        )
    if pr.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token expirado."
        )

    # Política
    ok, msg = password_policy_ok(payload.new_password)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    # Cambia password
    user = session.get(User, pr.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Usuario no encontrado."
        )

    user.hashed_password = hash_password(payload.new_password)
    pr.used_at = datetime.utcnow()

    session.add(user)
    session.add(pr)
    session.commit()
    return {"detail": "Contraseña restablecida."}

@router.post("/_dev/mail-test")
def mail_test(background: BackgroundTasks):
    html = "<h3>Prueba</h3><p>Correo funcionando 🚀</p>"
    background.add_task(
        send_email,
        to=settings.SMTP_USERNAME,  # te lo envías a ti mismo
        subject="Prueba SMTP Aura",
        html=html,
        text_fallback="Correo funcionando"
    )
    return {"detail": "Mail encolado"}