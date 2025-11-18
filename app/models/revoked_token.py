from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class RevokedToken(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    jti: str = Field(index=True, unique=True)  # identificador del JWT
    user_id: int = Field(index=True)
    expires_at: datetime
