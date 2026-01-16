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
    uploader: str  # 上传用户名
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

class ChatQueryLogOut(BaseModel):
    id: UUID
    query_time: datetime
    username: str
    query_content: str
    initial_rag_results: Optional[List[dict]] = None  # JSONB 字段，存储为列表
    reranked_results: Optional[List[dict]] = None  # JSONB 字段，存储为列表
    llm_response: Optional[str] = None
    model_name: Optional[str] = None
    query_duration_seconds: Optional[float] = None

class PaginatedChatLogs(BaseModel):
    total: int  # 总记录数
    page: int  # 当前页码
    page_size: int  # 每页大小
    total_pages: int  # 总页数
    items: List[ChatQueryLogOut]  # 当前页的数据

class LogQueryParams(BaseModel):
    page: int = 1  # 页码，从1开始
    page_size: int = 15  # 每页大小
    username: Optional[str] = None  # 可选：按用户名筛选
