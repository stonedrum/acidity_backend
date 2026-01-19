import os
# 必须在导入任何 huggingface 相关库之前设置环境变量，强制使用本地模型
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from .database import get_db, init_db
from .models import User, Document, Clause, ApiCallLog, ChatQueryLog, Prompt, DictType, DictData
from .schemas import (
    UserCreate, UserLogin, Token, DocumentOut, SearchQuery, ClauseOut, 
    ChatRequest, ChatQueryLogOut, PaginatedChatLogs, LogQueryParams, 
    PromptOut, PromptCreate, PromptUpdate, DictDataOut, DictTypeOut,
    DictTypeCreate, DictTypeUpdate, DictDataCreate, DictDataUpdate,
    ClauseCreate, ClauseUpdate, PaginatedClauses
)
from .auth import get_password_hash, verify_password, create_access_token, get_current_user
from .services.oss_service import oss_service
from .services.pdf_service import pdf_service
from .services.rag_service import rag_service
from .services.llm_service import llm_service
from .config import settings
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import shutil
import uuid
from uuid import UUID
import time
import json
from datetime import datetime
from typing import Optional, List

app = FastAPI(title="Standard Knowledge Base RAG")

@app.on_event("startup")
async def startup_event():
    await init_db()
    # 初始化默认提示词
    from .database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        # 检查是否存在 rag_system_prompt
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
        
        # --- 初始化知识库类型字典 ---
        # 检查是否已存在 kb_type 字典类型
        stmt = select(DictType).where(DictType.type_name == "kb_type")
        result = await db.execute(stmt)
        kb_type_dict = result.scalar_one_or_none()
        
        if not kb_type_dict:
            # 创建字典类型
            kb_type_dict = DictType(type_name="kb_type", description="知识库所属类型分类")
            db.add(kb_type_dict)
            await db.flush() # 获取 ID
            
            # 创建初始字典数据
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

async def get_prompt(db: AsyncSession, name: str, default_template: str = None) -> str:
    """从数据库获取提示词"""
    stmt = select(Prompt).where(Prompt.name == name, Prompt.is_active == True)
    result = await db.execute(stmt)
    prompt_obj = result.scalar_one_or_none()
    return prompt_obj.template if prompt_obj else default_template

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
        
        # 获取请求参数
        body = None
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                body_bytes = await request.body()
                if body_bytes:
                    try:
                        body = json.loads(body_bytes.decode())
                    except:
                        body = {"raw": body_bytes.decode()[:500]}  # 限制长度
        except:
            pass
        
        # 执行请求
        response = await call_next(request)
        
        # 计算响应时间
        response_time_ms = (time.time() - start_time) * 1000
        
        # 记录日志（异步，不阻塞响应）
        # 注意：中间件中直接访问数据库比较复杂，这里简化处理
        # 实际记录在接口层面完成，中间件主要用于统计
        pass
        
        return response

app.add_middleware(ApiLoggingMiddleware)

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
async def upload_pdf(
    file: UploadFile = File(...), 
    kb_type: str = Form(...),  # 接收知识库类型
    db: AsyncSession = Depends(get_db), 
    current_user: str = Depends(get_current_user)
):
    # 1. Save locally temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Generate UUID-based filename (remove hyphens and keep file extension)
        # 生成基于UUID的文件名（去掉横线，保留文件扩展名）
        file_ext = os.path.splitext(file.filename)[1]  # 获取文件扩展名，如 .pdf
        uuid_name = str(uuid.uuid4()).replace('-', '')  # 生成UUID并去掉横线
        oss_filename = f"{uuid_name}{file_ext}"  # 组合：uuid文件名 + 扩展名
        
        # 3. Upload to OSS (上传到 /laws/ 目录，使用UUID文件名)
        with open(temp_file_path, "rb") as f:
            oss_key = oss_service.upload_file(f.read(), oss_filename, directory="laws")
        
        # 4. Check if filename already exists (filename must be unique)
        # 检查文件名是否已存在（文件名必须唯一）
        existing_doc = await db.execute(select(Document).where(Document.filename == file.filename))
        if existing_doc.scalar_one_or_none():
            raise HTTPException(
                status_code=400, 
                detail=f"文件 '{file.filename}' 已存在，请使用不同的文件名"
            )
        
        # 5. Create Document record (保存原始文件名，OSS key使用UUID文件名，记录上传人，记录知识库类型)
        doc = Document(
            filename=file.filename, 
            oss_key=oss_key,
            uploader=current_user,  # 记录上传用户名
            kb_type=kb_type        # 记录知识库类型
        )
        db.add(doc)
        await db.flush() # Get doc.id
        
        # 6. Parse PDF
        clauses_data = pdf_service.parse_pdf(temp_file_path)
        
        # 7. Embedding and Save Clauses
        for item in clauses_data:
            embedding = rag_service.get_embedding(item["content"])
            clause = Clause(
                doc_id=doc.id,
                kb_type=kb_type, # 冗余存储类型
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
    results = await rag_service.search_and_rerank(
        query_data.query, 
        db, 
        kb_type=query_data.kb_type # 传递类型筛选
    )
    
    out = []
    for clause, score in results:
        out.append(ClauseOut(
            id=clause.id,
            kb_type=clause.kb_type,
            chapter_path=clause.chapter_path,
            content=clause.content,
            score=float(score)
        ))
    return out

@app.post("/chat")
async def chat(
    request: ChatRequest, 
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    # 记录查询开始时间
    query_start_time = time.time()
    username = current_user or "anonymous"
    
    # 获取模型名称
    model_name = request.model or settings.LLM_MODEL
    
    # 1. RAG: Retrieve relevant clauses (获取初始结果和重排结果)
    initial_results, reranked_results = await rag_service.search_and_rerank(
        request.message, 
        db, 
        kb_type=request.kb_type, # 传递类型筛选
        return_initial_results=True
    )
    
    # 使用重排结果构建上下文
    context = ""
    referenced_doc_ids = set()  # 收集引用的文档 ID
    results = []  # 用于后续处理
    for i, reranked_item in enumerate(reranked_results):
        # 从重排结果中获取完整的 clause 信息
        clause_id = reranked_item["clause_id"]
        stmt = select(Clause).where(Clause.id == clause_id)
        result = await db.execute(stmt)
        clause = result.scalar_one_or_none()
        if clause:
            context += f"【参考资料{i+1}】章节路径：{clause.chapter_path}\n内容：{clause.content}\n\n"
            referenced_doc_ids.add(clause.doc_id)
            results.append((clause, reranked_item["rerank_score"]))
    
    # 查询引用的文档信息
    referenced_docs = []
    if referenced_doc_ids:
        stmt = select(Document).where(Document.id.in_(referenced_doc_ids))
        result = await db.execute(stmt)
        referenced_docs = result.scalars().all()
    
    # 生成引用文件链接
    reference_links = ""
    if referenced_docs:
        reference_links = "\n\n---\n**📎 引用文件：**\n"
        for doc in referenced_docs:
            file_url = oss_service.get_file_url(doc.oss_key)
            reference_links += f"- [{doc.filename}]({file_url})\n"
    
    # 2. Prepare Prompt
    # 从数据库获取提示词模板，如果不存在则使用默认值
    default_system_template = """你是市政设施运维专家，精通结构健康监测、病害诊断、养护修复、应急处置及行业规范。请基于市政设施全生命周期运维经验，用专业、简洁的语言解答道桥隧巡检、维修、管理相关问题。
    将根据提供的【参考资料】来回答用户的问题。如果资料中没有相关信息，请诚实说明。
    你的回答应体现市政设施运维专家的身份：逻辑清晰、术语规范、强调安全与合规。

    重要提示：请不要在回答中包含引用文件、参考文献或链接信息，这些信息将由系统自动添加。

    【参考资料】
    {context}
"""
    prompt_template = await get_prompt(db, "rag_system_prompt", default_system_template)
    system_prompt = prompt_template.format(context=context)
    
    messages = [{"role": "system", "content": system_prompt}]
    # Add history
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
    # Add current message
    messages.append({"role": "user", "content": request.message})
    
    # 3. Call LLM
    if request.stream:
        async def stream_wrapper():
            collected_content = ""  # 收集所有内容
            try:
                print(f"[DEBUG] 开始调用 LLM，消息数量: {len(messages)}")
                # chat_completion 是 async 函数，返回异步生成器对象
                stream = await llm_service.chat_completion(messages, model=model_name, stream=True)
                print("[DEBUG] LLM 流式响应已建立，开始传输数据...")
                chunk_count = 0
                async for chunk in stream:
                    chunk_count += 1
                    if chunk_count <= 3:  # 只打印前3个chunk的调试信息
                        print(f"[DEBUG] 发送数据块 {chunk_count}: {chunk[:50]}...")
                    collected_content += chunk
                    yield chunk
                print(f"[DEBUG] 流式响应完成，共发送 {chunk_count} 个数据块")
                # 在流式响应结束时，检查是否已包含引用文件部分，避免重复添加
                if reference_links:
                    # 检查是否已经包含引用文件相关的标记
                    has_reference_section = (
                        "引用文件" in collected_content or 
                        "📎" in collected_content or
                        "参考文献" in collected_content.lower() or
                        "cited documents" in collected_content.lower()
                    )
                    if not has_reference_section:
                        yield reference_links
                        collected_content += reference_links
                    else:
                        print("[DEBUG] 检测到回答中已包含引用文件部分，跳过添加以避免重复")
            except Exception as e:
                # 如果发生错误，返回错误信息
                import traceback
                error_detail = str(e)
                print(f"[ERROR] 流式响应出错: {error_detail}")
                traceback.print_exc()
                error_msg = f"\n\n❌ 错误: {error_detail}\n\n请检查 API 密钥是否正确配置。"
                collected_content = error_msg
                yield error_msg
            finally:
                # 保存查询记录（异步，不阻塞响应）
                try:
                    query_duration = time.time() - query_start_time
                    # JSONB 可以直接存储 Python 字典
                    chat_log = ChatQueryLog(
                        username=username,
                        query_content=request.message,
                        initial_rag_results=initial_results,  # JSONB 直接存储字典
                        reranked_results=reranked_results,  # JSONB 直接存储字典
                        llm_response=collected_content,
                        model_name=model_name,
                        query_duration_seconds=query_duration
                    )
                    db.add(chat_log)
                    await db.commit()
                except Exception as e:
                    print(f"[ERROR] 保存查询记录失败: {e}")
        return StreamingResponse(stream_wrapper(), media_type="text/plain; charset=utf-8")
    else:
        try:
            response_content = await llm_service.chat_completion(messages, model=model_name, stream=False)
            # 在非流式响应中，检查是否已包含引用文件部分，避免重复添加
            if reference_links:
                # 检查是否已经包含引用文件相关的标记
                has_reference_section = (
                    "引用文件" in response_content or 
                    "📎" in response_content or
                    "参考文献" in response_content.lower() or
                    "cited documents" in response_content.lower()
                )
                if not has_reference_section:
                    response_content += reference_links
                else:
                    print("[DEBUG] 检测到回答中已包含引用文件部分，跳过添加以避免重复")
            
            # 保存查询记录
            try:
                query_duration = time.time() - query_start_time
                # JSONB 可以直接存储 Python 字典
                chat_log = ChatQueryLog(
                    username=username,
                    query_content=request.message,
                    initial_rag_results=initial_results,  # JSONB 直接存储字典
                    reranked_results=reranked_results,  # JSONB 直接存储字典
                    llm_response=response_content,
                    model_name=model_name,
                    query_duration_seconds=query_duration
                )
                db.add(chat_log)
                await db.commit()
            except Exception as e:
                print(f"[ERROR] 保存查询记录失败: {e}")
            
            return {"content": response_content}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM调用失败: {str(e)}")

@app.get("/chat-logs", response_model=PaginatedChatLogs)
async def get_chat_logs(
    page: int = 1,
    page_size: int = 15,
    username: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    获取 Chat 查询日志列表（分页）
    :param page: 页码，从1开始
    :param page_size: 每页大小，默认15
    :param username: 可选，按用户名筛选
    :param db: 数据库会话
    :param current_user: 当前用户（需要登录）
    """
    # 构建查询
    query = select(ChatQueryLog)
    
    # 如果指定了用户名，添加筛选条件
    if username:
        query = query.where(ChatQueryLog.username == username)
    
    # 按时间倒序排列（最新的在前）
    query = query.order_by(desc(ChatQueryLog.query_time))
    
    # 计算总数
    count_query = select(func.count()).select_from(ChatQueryLog)
    if username:
        count_query = count_query.where(ChatQueryLog.username == username)
    
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # 计算分页
    total_pages = (total + page_size - 1) // page_size  # 向上取整
    offset = (page - 1) * page_size
    
    # 获取当前页数据
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    logs = result.scalars().all()
    
    # 转换为输出格式
    items = []
    for log in logs:
        items.append(ChatQueryLogOut(
            id=log.id,
            query_time=log.query_time,
            username=log.username,
            query_content=log.query_content,
            initial_rag_results=log.initial_rag_results,
            reranked_results=log.reranked_results,
            llm_response=log.llm_response,
            model_name=log.model_name,
            query_duration_seconds=log.query_duration_seconds
        ))
    
    return PaginatedChatLogs(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        items=items
    )

# --- 提示词管理接口 ---

@app.get("/prompts", response_model=List[PromptOut])
async def list_prompts(
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """获取所有提示词"""
    stmt = select(Prompt).order_by(Prompt.name)
    result = await db.execute(stmt)
    return result.scalars().all()

@app.post("/prompts", response_model=PromptOut)
async def create_prompt(
    prompt_data: PromptCreate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """创建新提示词"""
    # 检查重名
    stmt = select(Prompt).where(Prompt.name == prompt_data.name)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"Prompt with name '{prompt_data.name}' already exists")
    
    new_prompt = Prompt(
        name=prompt_data.name,
        template=prompt_data.template,
        description=prompt_data.description,
        is_active=prompt_data.is_active
    )
    db.add(new_prompt)
    await db.commit()
    await db.refresh(new_prompt)
    return new_prompt

@app.put("/prompts/{prompt_id}", response_model=PromptOut)
async def update_prompt(
    prompt_id: UUID,
    prompt_data: PromptUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """更新提示词"""
    stmt = select(Prompt).where(Prompt.id == prompt_id)
    result = await db.execute(stmt)
    prompt_obj = result.scalar_one_or_none()
    if not prompt_obj:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    # 更新字段
    if prompt_data.name is not None:
        # 如果改名，检查是否冲突
        if prompt_data.name != prompt_obj.name:
            name_check = await db.execute(select(Prompt).where(Prompt.name == prompt_data.name))
            if name_check.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="New name already exists")
        prompt_obj.name = prompt_data.name
    
    if prompt_data.template is not None:
        prompt_obj.template = prompt_data.template
        prompt_obj.version += 1  # 每次修改模板增加版本号
    
    if prompt_data.description is not None:
        prompt_obj.description = prompt_data.description
        
    if prompt_data.is_active is not None:
        prompt_obj.is_active = prompt_data.is_active
    
    prompt_obj.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(prompt_obj)
    return prompt_obj

@app.delete("/prompts/{prompt_id}")
async def delete_prompt(
    prompt_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """删除提示词"""
    stmt = select(Prompt).where(Prompt.id == prompt_id)
    result = await db.execute(stmt)
    prompt_obj = result.scalar_one_or_none()
    if not prompt_obj:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    # 禁止删除核心系统提示词
    if prompt_obj.name == "rag_system_prompt":
        raise HTTPException(status_code=400, detail="Cannot delete core system prompt")
    
    await db.delete(prompt_obj)
    await db.commit()
    return {"message": "Prompt deleted successfully"}

# --- 数据字典接口 ---

@app.get("/dicts/{type_name}", response_model=List[DictDataOut])
async def get_dict_by_type(type_name: str, db: AsyncSession = Depends(get_db)):
    """根据类型名称获取字典项（如 /dicts/kb_type）"""
    stmt = (
        select(DictData)
        .join(DictType)
        .where(DictType.type_name == type_name, DictData.is_active == True)
        .order_by(DictData.sort_order)
    )
    result = await db.execute(stmt)
    return result.scalars().all()

@app.get("/dict-types", response_model=List[DictTypeOut])
async def list_dict_types(db: AsyncSession = Depends(get_db), current_user: str = Depends(get_current_user)):
    """列出所有字典类型（管理端使用）"""
    stmt = select(DictType)
    result = await db.execute(stmt)
    types = result.scalars().all()
    
    # 手动加载关联数据，避免 N+1 或延迟加载问题
    out = []
    for t in types:
        data_stmt = select(DictData).where(DictData.type_id == t.id).order_by(DictData.sort_order)
        data_result = await db.execute(data_stmt)
        t_data = data_result.scalars().all()
        out.append(DictTypeOut(
            id=t.id,
            type_name=t.type_name,
            description=t.description,
            data=[DictDataOut.from_orm(d) for d in t_data]
        ))
    return out

@app.post("/dict-types", response_model=DictTypeOut)
async def create_dict_type(
    type_data: DictTypeCreate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """创建字典类型"""
    stmt = select(DictType).where(DictType.type_name == type_data.type_name)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Dict type already exists")
    
    new_type = DictType(**type_data.dict())
    db.add(new_type)
    await db.commit()
    await db.refresh(new_type)
    return DictTypeOut(id=new_type.id, type_name=new_type.type_name, description=new_type.description, data=[])

@app.put("/dict-types/{type_id}", response_model=DictTypeOut)
async def update_dict_type(
    type_id: UUID,
    type_data: DictTypeUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """更新字典类型"""
    stmt = select(DictType).where(DictType.id == type_id)
    result = await db.execute(stmt)
    dict_type = result.scalar_one_or_none()
    if not dict_type:
        raise HTTPException(status_code=404, detail="Dict type not found")
    
    for key, value in type_data.dict(exclude_unset=True).items():
        setattr(dict_type, key, value)
    
    await db.commit()
    await db.refresh(dict_type)
    
    # 加载数据项
    data_stmt = select(DictData).where(DictData.type_id == dict_type.id).order_by(DictData.sort_order)
    data_result = await db.execute(data_stmt)
    t_data = data_result.scalars().all()
    
    return DictTypeOut(
        id=dict_type.id,
        type_name=dict_type.type_name,
        description=dict_type.description,
        data=[DictDataOut.from_orm(d) for d in t_data]
    )

@app.delete("/dict-types/{type_id}")
async def delete_dict_type(
    type_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """删除字典类型"""
    stmt = select(DictType).where(DictType.id == type_id)
    result = await db.execute(stmt)
    dict_type = result.scalar_one_or_none()
    if not dict_type:
        raise HTTPException(status_code=404, detail="Dict type not found")
    
    # 核心字典类型禁止删除
    if dict_type.type_name == "kb_type":
        raise HTTPException(status_code=400, detail="Core dict type 'kb_type' cannot be deleted")
        
    await db.delete(dict_type)
    await db.commit()
    return {"message": "Dict type deleted"}

# --- 字典数据项 CRUD ---

@app.post("/dict-data/{type_id}", response_model=DictDataOut)
async def create_dict_data(
    type_id: UUID,
    data_item: DictDataCreate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """创建字典数据项"""
    # 检查类型是否存在
    stmt = select(DictType).where(DictType.id == type_id)
    result = await db.execute(stmt)
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Dict type not found")
        
    new_data = DictData(type_id=type_id, **data_item.dict())
    db.add(new_data)
    await db.commit()
    await db.refresh(new_data)
    return new_data

@app.put("/dict-data/{data_id}", response_model=DictDataOut)
async def update_dict_data(
    data_id: UUID,
    data_item: DictDataUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """更新字典数据项"""
    stmt = select(DictData).where(DictData.id == data_id)
    result = await db.execute(stmt)
    db_data = result.scalar_one_or_none()
    if not db_data:
        raise HTTPException(status_code=404, detail="Dict data not found")
    
    for key, value in data_item.dict(exclude_unset=True).items():
        setattr(db_data, key, value)
    
    await db.commit()
    await db.refresh(db_data)
    return db_data

@app.delete("/dict-data/{data_id}")
async def delete_dict_data(
    data_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """删除字典数据项"""
    stmt = select(DictData).where(DictData.id == data_id)
    result = await db.execute(stmt)
    db_data = result.scalar_one_or_none()
    if not db_data:
        raise HTTPException(status_code=404, detail="Dict data not found")
        
    await db.delete(db_data)
    await db.commit()
    return {"message": "Dict data deleted"}

# --- 知识条款管理接口 ---

@app.get("/clauses", response_model=PaginatedClauses)
async def list_clauses(
    page: int = 1,
    page_size: int = 15,
    kb_type: Optional[str] = None,
    doc_id: Optional[UUID] = None,
    keyword: Optional[str] = None,
    is_verified: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """分页获取知识条款列表"""
    stmt = select(Clause)
    if kb_type:
        stmt = stmt.where(Clause.kb_type == kb_type)
    if doc_id:
        stmt = stmt.where(Clause.doc_id == doc_id)
    if keyword:
        stmt = stmt.where(Clause.content.ilike(f"%{keyword}%") | Clause.chapter_path.ilike(f"%{keyword}%"))
    if is_verified is not None:
        stmt = stmt.where(Clause.is_verified == is_verified)
    
    # 计算总数
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_res = await db.execute(count_stmt)
    total = total_res.scalar()
    
    # 分页排序：
    # 1. 优先显示未校验 (is_verified 为 False 的排在前面)
    # 2. 其次按章节路径排序
    # 3. 最后按 ID 排序（保底，确保编辑后物理位置变了但逻辑排序不变）
    stmt = stmt.order_by(
        Clause.is_verified.asc(), 
        Clause.chapter_path.asc(), 
        Clause.id.asc()
    ).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(stmt)
    clauses = result.scalars().all()
    
    # 获取文档名称映射
    doc_ids = list(set(c.doc_id for c in clauses if c.doc_id))
    doc_map = {}
    if doc_ids:
        doc_stmt = select(Document.id, Document.filename).where(Document.id.in_(doc_ids))
        doc_res = await db.execute(doc_stmt)
        doc_map = {row.id: row.filename for row in doc_res}
    
    return PaginatedClauses(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
        items=[
            ClauseOut(
                id=c.id,
                kb_type=c.kb_type,
                chapter_path=c.chapter_path,
                content=c.content,
                is_verified=c.is_verified, # 确保返回真实的校验状态
                doc_id=c.doc_id, # 返回文档 ID
                doc_name=doc_map.get(c.doc_id, "手动新增") # 返回文档名称
            ) for c in clauses
        ]
    )

@app.post("/clauses", response_model=ClauseOut)
async def create_clause(
    data: ClauseCreate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """手动创建知识条款"""
    embedding = rag_service.get_embedding(data.content)
    new_clause = Clause(
        kb_type=data.kb_type,
        chapter_path=data.chapter_path,
        content=data.content,
        doc_id=data.doc_id,
        embedding=embedding,
        is_verified=True # 手动创建的默认为已校验
    )
    db.add(new_clause)
    await db.commit()
    await db.refresh(new_clause)
    
    # 获取文档名称
    doc_name = "手动新增"
    if new_clause.doc_id:
        doc_stmt = select(Document.filename).where(Document.id == new_clause.doc_id)
        doc_res = await db.execute(doc_stmt)
        doc_name = doc_res.scalar() or "手动新增"

    return ClauseOut(
        id=new_clause.id,
        kb_type=new_clause.kb_type,
        chapter_path=new_clause.chapter_path,
        content=new_clause.content,
        is_verified=new_clause.is_verified,
        doc_id=new_clause.doc_id,
        doc_name=doc_name
    )

@app.put("/clauses/{clause_id}", response_model=ClauseOut)
async def update_clause(
    clause_id: UUID,
    data: ClauseUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """更新知识条款"""
    stmt = select(Clause).where(Clause.id == clause_id)
    result = await db.execute(stmt)
    clause = result.scalar_one_or_none()
    if not clause:
        raise HTTPException(status_code=404, detail="Clause not found")
    
    # 如果修改了内容，需要重新生成向量
    if data.content is not None and data.content != clause.content:
        clause.embedding = rag_service.get_embedding(data.content)
        clause.content = data.content
    
    if data.kb_type is not None:
        clause.kb_type = data.kb_type
    if data.chapter_path is not None:
        clause.chapter_path = data.chapter_path
    if data.is_verified is not None:
        clause.is_verified = data.is_verified
    if data.doc_id is not None:
        clause.doc_id = data.doc_id
        
    await db.commit()
    await db.refresh(clause)
    
    # 获取文档名称
    doc_name = "手动新增"
    if clause.doc_id:
        doc_stmt = select(Document.filename).where(Document.id == clause.doc_id)
        doc_res = await db.execute(doc_stmt)
        doc_name = doc_res.scalar() or "手动新增"

    return ClauseOut(
        id=clause.id,
        kb_type=clause.kb_type,
        chapter_path=clause.chapter_path,
        content=clause.content,
        is_verified=clause.is_verified,
        doc_id=clause.doc_id,
        doc_name=doc_name
    )

@app.delete("/clauses/{clause_id}")
async def delete_clause(
    clause_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """删除知识条款"""
    stmt = select(Clause).where(Clause.id == clause_id)
    result = await db.execute(stmt)
    clause = result.scalar_one_or_none()
    if not clause:
        raise HTTPException(status_code=404, detail="Clause not found")
        
    await db.delete(clause)
    await db.commit()
    return {"message": "Clause deleted"}

# --- 文档管理接口 ---

@app.get("/documents", response_model=List[DocumentOut])
async def list_documents(
    keyword: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """获取所有文档列表（支持模糊匹配文件名）"""
    stmt = select(Document)
    if keyword:
        stmt = stmt.where(Document.filename.ilike(f"%{keyword}%"))
    stmt = stmt.order_by(Document.upload_time.desc())
    result = await db.execute(stmt)
    return result.scalars().all()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
