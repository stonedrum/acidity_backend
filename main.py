import os
# 必须在导入任何 huggingface 相关库之前设置环境变量，强制使用本地模型
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .database import get_db, init_db
from .models import User, Document, Clause
from .schemas import UserCreate, UserLogin, Token, DocumentOut, SearchQuery, ClauseOut, ChatRequest
from .auth import get_password_hash, verify_password, create_access_token, get_current_user
from .services.oss_service import oss_service
from .services.pdf_service import pdf_service
from .services.rag_service import rag_service
from .services.llm_service import llm_service
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid

app = FastAPI(title="Standard Knowledge Base RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await init_db()

@app.post("/register", response_model=Token)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    db_user = await db.execute(select(User).where(User.username == user_data.username))
    if db_user.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user_data.password)
    new_user = User(username=user_data.username, password_hash=hashed_password)
    db.add(new_user)
    await db.commit()
    
    access_token = create_access_token(data={"sub": new_user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token", response_model=Token)
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == user_data.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(user_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), db: AsyncSession = Depends(get_db), current_user: str = Depends(get_current_user)):
    # 1. Save locally temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Upload to OSS
        with open(temp_file_path, "rb") as f:
            oss_key = oss_service.upload_file(f.read(), file.filename)
        
        # 3. Create Document record
        doc = Document(filename=file.filename, oss_key=oss_key)
        db.add(doc)
        await db.flush() # Get doc.id
        
        # 4. Parse PDF
        clauses_data = pdf_service.parse_pdf(temp_file_path)
        
        # 5. Embedding and Save Clauses
        for item in clauses_data:
            embedding = rag_service.get_embedding(item["content"])
            clause = Clause(
                doc_id=doc.id,
                chapter_path=item["chapter_path"],
                content=item["content"],
                embedding=embedding
            )
            db.add(clause)
        
        await db.commit()
        return {"message": "Upload and processing successful", "doc_id": doc.id}
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/search", response_model=list[ClauseOut])
async def search(query_data: SearchQuery, db: AsyncSession = Depends(get_db)):
    results = await rag_service.search_and_rerank(query_data.query, db)
    
    out = []
    for clause, score in results:
        out.append(ClauseOut(
            id=clause.id,
            chapter_path=clause.chapter_path,
            content=clause.content,
            score=float(score)
        ))
    return out

@app.post("/chat")
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    # 1. RAG: Retrieve relevant clauses
    results = await rag_service.search_and_rerank(request.message, db)
    context = ""
    for i, (clause, score) in enumerate(results):
        context += f"【参考资料{i+1}】章节路径：{clause.chapter_path}\n内容：{clause.content}\n\n"
    
    # 2. Prepare Prompt
    system_prompt = f"""你是一位资深的工程专家。你佩戴安全帽，穿着反光背心，戴着眼镜，显得专业、严谨且富有经验。
你将根据提供的【参考资料】来回答用户的问题。如果资料中没有相关信息，请诚实说明。
你的回答应体现工程专家的身份：逻辑清晰、术语规范、强调安全与合规。

【参考资料】
{context}
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    # Add history
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
    # Add current message
    messages.append({"role": "user", "content": request.message})
    
    # 3. Call LLM
    if request.stream:
        async def stream_wrapper():
            try:
                print(f"[DEBUG] 开始调用 LLM，消息数量: {len(messages)}")
                # chat_completion 是 async 函数，返回异步生成器对象
                stream = await llm_service.chat_completion(messages, model=request.model, stream=True)
                print("[DEBUG] LLM 流式响应已建立，开始传输数据...")
                chunk_count = 0
                async for chunk in stream:
                    chunk_count += 1
                    if chunk_count <= 3:  # 只打印前3个chunk的调试信息
                        print(f"[DEBUG] 发送数据块 {chunk_count}: {chunk[:50]}...")
                    yield chunk
                print(f"[DEBUG] 流式响应完成，共发送 {chunk_count} 个数据块")
            except Exception as e:
                # 如果发生错误，返回错误信息
                import traceback
                error_detail = str(e)
                print(f"[ERROR] 流式响应出错: {error_detail}")
                traceback.print_exc()
                error_msg = f"\n\n❌ 错误: {error_detail}\n\n请检查 API 密钥是否正确配置。"
                yield error_msg
        return StreamingResponse(stream_wrapper(), media_type="text/plain; charset=utf-8")
    else:
        try:
            response_content = await llm_service.chat_completion(messages, model=request.model, stream=False)
            return {"content": response_content}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM调用失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
