# app/core/email.py
import ssl
import smtplib
from email.message import EmailMessage
from typing import Optional
from app.core.config import settings

def _validate_email_settings():
    if not settings.SMTP_HOST:
        raise RuntimeError("SMTP_HOST no configurado")
    if settings.SMTP_PORT is None:
        raise RuntimeError("SMTP_PORT no configurado")
    if not settings.SMTP_USERNAME:
        raise RuntimeError("SMTP_USERNAME no configurado")
    if not settings.SMTP_PASSWORD:
        raise RuntimeError("SMTP_PASSWORD no configurado")
    if not settings.EMAIL_FROM:
        # fallback razonable: usar el username como from
        settings.EMAIL_FROM = settings.SMTP_USERNAME  # type: ignore

def send_email(to: str, subject: str, html: str, text_fallback: Optional[str] = None):
    _validate_email_settings()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.EMAIL_FROM
    msg["To"] = to
    if text_fallback:
        msg.set_content(text_fallback)
    msg.add_alternative(html, subtype="html")

    port = int(settings.SMTP_PORT)  # ya validado que no es None
    host = settings.SMTP_HOST

    if port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.send_message(msg)
    else:
        # típico 587 (STARTTLS) u otros
        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            try:
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            except smtplib.SMTPNotSupportedError:
                # si el servidor no soporta STARTTLS, seguimos plano
                pass
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.send_message(msg)
