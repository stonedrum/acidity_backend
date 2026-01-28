from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        # Create extension if not exists
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        
        # 兼容性处理：如果 clauses 表没有相应字段，则添加
        try:
            await conn.execute(text("ALTER TABLE clauses ADD COLUMN IF NOT EXISTS page_number INTEGER"))
            await conn.execute(text("ALTER TABLE clauses ADD COLUMN IF NOT EXISTS creator VARCHAR"))
            await conn.execute(text("ALTER TABLE clauses ADD COLUMN IF NOT EXISTS import_method VARCHAR"))
            await conn.execute(text("ALTER TABLE clauses ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
            await conn.execute(text("ALTER TABLE clauses ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
            await conn.execute(text("ALTER TABLE clauses ADD COLUMN IF NOT EXISTS updated_by VARCHAR"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_clauses_creator ON clauses (creator)"))
        except Exception as e:
            print(f"[init_db] 尝试添加字段失败: {e}")
