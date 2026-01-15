from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
from datetime import datetime

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class UserLogin(UserBase):
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class DocumentOut(BaseModel):
    id: UUID
    filename: str
    upload_time: datetime

class ClauseOut(BaseModel):
    id: UUID
    chapter_path: str
    content: str
    score: Optional[float] = None

class SearchQuery(BaseModel):
    query: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    stream: Optional[bool] = False
    model: Optional[str] = None
