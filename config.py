import os
from pydantic_settings import BaseSettings

# Get the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/aicity"
    
    # Aliyun OSS
    OSS_ACCESS_KEY_ID: str = ""
    OSS_ACCESS_KEY_SECRET: str = ""
    OSS_BUCKET_NAME: str = ""
    OSS_ENDPOINT: str = "oss-cn-hangzhou.aliyuncs.com"
    
    # Models
    VECTOR_MODEL_NAME: str = "shibing624/text2vec-base-chinese"
    VECTOR_DIMENSION: int = 768
    RERANK_MODEL_NAME: str = "BAAI/bge-reranker-large"
    
    # LLM (Aliyun DashScope / OpenAI compatible)
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "qwen-plus"
    
    # JWT
    SECRET_KEY: str = "csj1234567890"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    model_config = {
        "env_file": ENV_FILE,
        "extra": "ignore"
    }

settings = Settings()
