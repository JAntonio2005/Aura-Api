from pydantic import BaseModel, EmailStr, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: str | None = None

class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str | None = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictionLogOut(BaseModel):
    id: int
    image_name: Optional[str]
    top1_label: str
    top1_score: float
    top5: List[Dict[str, Any]]
    created_at: datetime

class SearchLogOut(BaseModel):
    id: int
    query_label: str
    created_at: datetime

    
class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str

class ForgotPasswordIn(BaseModel):
    email: EmailStr

class ResetPasswordIn(BaseModel):
    token: str
    new_password: str