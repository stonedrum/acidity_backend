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
    # 模型路径：如果是本地路径（以 / 或 ./ 开头），则直接使用；否则为 HuggingFace 模型名称
    # 默认情况下，sentence-transformers 会将模型缓存在 ~/.cache/huggingface/hub/
    VECTOR_MODEL_NAME: str = "shibing624/text2vec-base-chinese"  # 或本地路径，如 "/path/to/models/text2vec-base-chinese"
    VECTOR_MODEL_PATH: str = ""  # 自定义向量模型路径，如果设置则优先使用此路径
    VECTOR_DIMENSION: int = 768
    RERANK_MODEL_NAME: str = "BAAI/bge-reranker-large"  # 或本地路径，如 "/path/to/models/bge-reranker-large"
    RERANK_MODEL_PATH: str = ""  # 自定义重排模型路径，如果设置则优先使用此路径
    
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
