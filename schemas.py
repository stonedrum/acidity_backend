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
    kb_type: Optional[str] = None  # 知识库类型
    upload_time: datetime

class ClauseOut(BaseModel):
    id: UUID
    kb_type: Optional[str] = None
    chapter_path: str
    content: str
    is_verified: bool = False
    doc_id: Optional[UUID] = None
    doc_name: Optional[str] = None
    score: Optional[float] = None

class SearchQuery(BaseModel):
    query: str
    kb_type: Optional[str] = None  # 可选：按类型筛选

# --- 知识条款管理相关 Schema ---

class ClauseCreate(BaseModel):
    kb_type: str
    chapter_path: str
    content: str
    doc_id: Optional[UUID] = None

class ClauseBatchItem(BaseModel):
    chapter_path: str
    content: str

class ClauseBatchCreate(BaseModel):
    kb_type: str
    doc_id: Optional[UUID] = None
    items: List[ClauseBatchItem]

class BatchInsertResult(BaseModel):
    inserted: int

class ClauseUpdate(BaseModel):
    kb_type: Optional[str] = None
    chapter_path: Optional[str] = None
    content: Optional[str] = None
    is_verified: Optional[bool] = None
    doc_id: Optional[UUID] = None

class ClauseBatchUpdate(BaseModel):
    ids: List[UUID]
    kb_type: Optional[str] = None
    doc_id: Optional[UUID] = None
    is_verified: Optional[bool] = None

class PaginatedClauses(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    items: List[ClauseOut]

class DocumentOut(BaseModel):
    id: UUID
    filename: str
    uploader: str
    kb_type: Optional[str] = None
    upload_time: datetime
    file_url: Optional[str] = None

    class Config:
        from_attributes = True

class PaginatedDocuments(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    items: List[DocumentOut]

class DocumentCreate(BaseModel):
    filename: str
    kb_type: str
    oss_key: Optional[str] = None

class DocumentUpdate(BaseModel):
    filename: Optional[str] = None
    kb_type: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    stream: Optional[bool] = False
    model: Optional[str] = None
    kb_type: Optional[str] = None  # 可选：按类型筛选

class ChatQueryLogOut(BaseModel):
    id: UUID
    query_time: datetime
    username: str
    query_content: str
    initial_rag_results: Optional[List[dict]] = None  # JSONB 字段，存储为列表
    reranked_results: Optional[List[dict]] = None  # JSONB 字段，存储为列表
    llm_response: Optional[str] = None
    llm_messages: Optional[List[dict]] = None  # 包含完整的 Prompt 上下文
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

class PromptOut(BaseModel):
    id: UUID
    name: str
    template: str
    is_active: bool
    version: int
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PromptCreate(BaseModel):
    name: str
    template: str
    description: Optional[str] = None
    is_active: Optional[bool] = True

class PromptUpdate(BaseModel):
    name: Optional[str] = None
    template: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None

# --- 数据字典相关 Schema ---

class DictDataOut(BaseModel):
    id: UUID
    label: str
    value: str
    sort_order: int
    is_active: bool

    class Config:
        from_attributes = True

class DictTypeOut(BaseModel):
    id: UUID
    type_name: str
    description: Optional[str]
    data: List[DictDataOut] = []

    class Config:
        from_attributes = True

class DictDataCreate(BaseModel):
    label: str
    value: str
    sort_order: Optional[int] = 0
    is_active: Optional[bool] = True

class DictDataUpdate(BaseModel):
    label: Optional[str] = None
    value: Optional[str] = None
    sort_order: Optional[int] = None
    is_active: Optional[bool] = None

class DictTypeCreate(BaseModel):
    type_name: str
    description: Optional[str] = None

class DictTypeUpdate(BaseModel):
    type_name: Optional[str] = None
    description: Optional[str] = None
