# app/db.py
from sqlmodel import SQLModel, create_engine, Session
from app.core.config import settings

# Importa modelos para que SQLModel registre tablas:
from app.models import user as user_model       # noqa: F401
from app.models import history as history_model # noqa: F401
from app.models import password_reset as pr_model # <- si tienes el modelo de reset   # noqa: F401
from app.models import revoked_token as revoked_token_model 
from app.models import breed as breed_model
engine = create_engine(
    settings.DB_URL,
    echo=False,
    pool_pre_ping=True,      # evita conexiones muertas
    # connect_args={"sslmode": "require"}  # solo si no lo pusiste en el URL
)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
