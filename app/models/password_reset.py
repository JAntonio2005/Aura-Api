from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class PasswordReset(SQLModel, table=True):
    __tablename__ = "password_resets"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
    token: str = Field(index=True, max_length=255, unique=True)
    expires_at: datetime = Field(index=True)
    used_at: Optional[datetime] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
