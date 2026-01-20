import os
import time
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from sqlalchemy import select

# 必须在导入任何 huggingface 相关库之前设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from .database import init_db, AsyncSessionLocal
from .models import Prompt, DictType, DictData
from .config import settings
from .routers import auth, documents, clauses, chat, prompts, dicts

app = FastAPI(title="Standard Knowledge Base RAG")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 接口调用日志中间件
class ApiLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 获取用户名（从token中解析，如果存在）
        username = None
        try:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                from jose import jwt
                try:
                    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
                    username = payload.get("sub")
                except:
                    pass
        except:
            pass
        
        # 执行请求
        response = await call_next(request)
        
        # 计算响应时间
        response_time_ms = (time.time() - start_time) * 1000
        
        # TODO: 可以在这里记录 ApiCallLog 到数据库
        # 注意：中间件中直接异步访问数据库需要处理好连接池
        
        return response

app.add_middleware(ApiLoggingMiddleware)

@app.on_event("startup")
async def startup_event():
    # 初始化数据库表
    await init_db()
    
    # 初始化默认数据
    async with AsyncSessionLocal() as db:
        # 1. 初始化默认提示词
        stmt = select(Prompt).where(Prompt.name == "rag_system_prompt")
        result = await db.execute(stmt)
        if not result.scalar_one_or_none():
            default_prompt = """你是市政设施运维专家，精通结构健康监测、病害诊断、养护修复、应急处置及行业规范。请基于市政设施全生命周期运维经验，用专业、简洁的语言解答道桥隧巡检、维修、管理相关问题。
将根据提供的【参考资料】来回答用户的问题。如果资料中没有相关信息，请诚实说明。
你的回答应体现市政设施运维专家的身份：逻辑清晰、术语规范、强调安全与合规。

重要提示：请不要在回答中包含引用文件、参考文献或链接信息，这些信息将由系统自动添加。

【参考资料】
{context}
"""
            new_prompt = Prompt(
                name="rag_system_prompt",
                template=default_prompt,
                description="RAG 系统的核心提示词模板"
            )
            db.add(new_prompt)
            await db.commit()
            print("[INFO] 已初始化默认提示词：rag_system_prompt")
        
        # 2. 初始化知识库类型字典
        stmt = select(DictType).where(DictType.type_name == "kb_type")
        result = await db.execute(stmt)
        kb_type_dict = result.scalar_one_or_none()
        
        if not kb_type_dict:
            kb_type_dict = DictType(type_name="kb_type", description="知识库所属类型分类")
            db.add(kb_type_dict)
            await db.flush()
            
            default_data = [
                {"label": "桥梁", "value": "bridge", "sort": 1},
                {"label": "道路", "value": "road", "sort": 2},
                {"label": "隧道", "value": "tunnel", "sort": 3},
                {"label": "公园绿化", "value": "park", "sort": 4},
                {"label": "排水", "value": "drainage", "sort": 5}
            ]
            for item in default_data:
                db.add(DictData(
                    type_id=kb_type_dict.id,
                    label=item["label"],
                    value=item["value"],
                    sort_order=item["sort"]
                ))
            await db.commit()
            print("[INFO] 已初始化数据字典：kb_type")

# 注册路由
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(clauses.router)
app.include_router(chat.router)
app.include_router(prompts.router)
app.include_router(dicts.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
