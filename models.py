import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Float, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
from .config import settings
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String, default="user") # admin, user

class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, unique=True, index=True)  # 文件名唯一
    oss_key = Column(String)
    uploader = Column(String)  # 上传用户名
    kb_type = Column(String, index=True, comment="知识库类型：桥梁、道路、隧道等")  # 增加知识库类型
    upload_time = Column(DateTime, default=datetime.utcnow)
    
    clauses = relationship("Clause", back_populates="document", cascade="all, delete-orphan")

class Clause(Base):
    __tablename__ = "clauses"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
    kb_type = Column(String, index=True, comment="冗余存储知识库类型，用于快速搜索过滤")  # 冗余存储类型
    chapter_path = Column(String) # e.g. "第一章 > 第1.1节"
    content = Column(Text) # Markdown content
    embedding = Column(Vector(settings.VECTOR_DIMENSION))
    is_verified = Column(Boolean, default=False)  # 是否已校验

    document = relationship("Document", back_populates="clauses")

class ApiCallLog(Base):
    """接口调用日志表"""
    __tablename__ = "api_call_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_time = Column(DateTime, default=datetime.utcnow, index=True)  # 调用时间
    username = Column(String, index=True)  # 用户名
    api_name = Column(String, index=True)  # 接口名称，如 "/chat", "/upload", "/search"
    method = Column(String)  # HTTP方法，如 "POST", "GET"
    parameters = Column(JSONB)  # 请求参数（JSON格式）
    response_status = Column(Integer)  # 响应状态码
    response_time_ms = Column(Float)  # 响应时间（毫秒）

class ChatQueryLog(Base):
    """Chat查询记录表"""
    __tablename__ = "chat_query_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_time = Column(DateTime, default=datetime.utcnow, index=True)  # 查询时间
    username = Column(String, index=True)  # 用户名
    query_content = Column(Text)  # 查询内容
    # 第一次从RAG数据库返回的记录（向量匹配的Top 10）
    initial_rag_results = Column(JSONB)  # 格式: [{"clause_id": "...", "chapter_path": "...", "content": "...", "score": 0.5}, ...]
    # 第二次重排后的信息（Top 3）
    reranked_results = Column(JSONB)  # 格式: [{"clause_id": "...", "chapter_path": "...", "content": "...", "rerank_score": 0.8}, ...]
    # 最终从大模型返回的内容
    llm_response = Column(Text)  # LLM返回的完整内容
    llm_messages = Column(JSONB)  # 发送给 LLM 的完整消息列表（包含 system, history, user）
    model_name = Column(String)  # 调用的模型名称
    query_duration_seconds = Column(Float)  # 查询所花时间（秒）

class Prompt(Base):
    """提示词配置表"""
    __tablename__ = "prompts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, index=True)  # 提示词名称，如 "rag_system_prompt"
    template = Column(Text, nullable=False)  # 提示词模板
    is_active = Column(Boolean, default=True)  # 是否启用
    version = Column(Integer, default=1)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DictType(Base):
    """字典类型表 (如 kb_type)"""
    __tablename__ = "dict_types"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type_name = Column(String, unique=True, index=True) # 如 "kb_type"
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class DictData(Base):
    """字典数据表 (如 桥梁, 道路)"""
    __tablename__ = "dict_data"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type_id = Column(UUID(as_uuid=True), ForeignKey("dict_types.id", ondelete="CASCADE"))
    label = Column(String, nullable=False) # 展示名称，如 "桥梁"
    value = Column(String, nullable=False) # 存储值，如 "bridge"
    sort_order = Column(Integer, default=0) # 排序号
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelComparisonVote(Base):
    """模型比对投票记录表"""
    __tablename__ = "model_comparison_votes"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vote_time = Column(DateTime, default=datetime.utcnow, index=True)
    username = Column(String, index=True)
    query_content = Column(Text)
    qwen_response = Column(Text)
    deepseek_response = Column(Text)
    winner = Column(Integer)  # 1 为 qwen-plus, 2 为 deepseek-v3.2
    llm_messages = Column(JSONB) # 新增：存储当时比对时的上下文信息

class SystemConfig(Base):
    """系统配置表"""
    __tablename__ = "system_configs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config_key = Column(String, unique=True, index=True) # 如 "llm_api_key"
    config_value = Column(Text)
    description = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
