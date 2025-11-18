from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, UniqueConstraint

class User(SQLModel, table=True):
    __tablename__ = "user"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True)
    hashed_password: str
    full_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    token_version: int = Field(default=0) 