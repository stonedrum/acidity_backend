import time
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from ..database import get_db
from ..models import Document, Clause, ChatQueryLog
from ..schemas import (
    SearchQuery, ClauseOut, ChatRequest, ChatQueryLogOut, PaginatedChatLogs
)
from ..auth import get_current_user
from ..services.rag_service import rag_service
from ..services.llm_service import llm_service
from ..services.oss_service import oss_service
from ..services.prompt_service import get_prompt_template
from ..config import settings

router = APIRouter(tags=["问答检索"])

@router.post("/search", response_model=List[ClauseOut])
async def search(query_data: SearchQuery, db: AsyncSession = Depends(get_db)):
    results = await rag_service.search_and_rerank(
        query_data.query, 
        db, 
        kb_type=query_data.kb_type
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

@router.post("/chat")
async def chat(
    request: ChatRequest, 
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    query_start_time = time.time()
    username = current_user or "anonymous"
    model_name = request.model or settings.LLM_MODEL
    
    # 1. RAG
    initial_results, reranked_results = await rag_service.search_and_rerank(
        request.message, 
        db, 
        kb_type=request.kb_type,
        return_initial_results=True
    )
    
    context = ""
    referenced_doc_ids = set()
    results = []
    for i, reranked_item in enumerate(reranked_results):
        clause_id = reranked_item["clause_id"]
        stmt = select(Clause).where(Clause.id == clause_id)
        result = await db.execute(stmt)
        clause = result.scalar_one_or_none()
        if clause:
            context += f"【参考资料{i+1}】章节路径：{clause.chapter_path}\n内容：{clause.content}\n\n"
            referenced_doc_ids.add(clause.doc_id)
            results.append((clause, reranked_item["rerank_score"]))
    
    referenced_docs = []
    if referenced_doc_ids:
        stmt = select(Document).where(Document.id.in_(referenced_doc_ids))
        result = await db.execute(stmt)
        referenced_docs = result.scalars().all()
    
    reference_links = ""
    if referenced_docs:
        reference_links = "\n\n---\n**📎 引用文件：**\n"
        for doc in referenced_docs:
            file_url = oss_service.get_file_url(doc.oss_key)
            reference_links += f"- [{doc.filename}]({file_url})\n"
    
    # 2. Prepare Prompt
    default_system_template = """你是市政设施运维专家，精通结构健康监测、病害诊断、养护修复、应急处置及行业规范。请基于市政设施全生命周期运维经验，用专业、简洁的语言解答道桥隧巡检、维修、管理相关问题。
    将根据提供的【参考资料】来回答用户的问题。如果资料中没有相关信息，请诚实说明。
    你的回答应体现市政设施运维专家的身份：逻辑清晰、术语规范、强调安全与合规。

    重要提示：请不要在回答中包含引用文件、参考文献或链接信息，这些信息将由系统自动添加。

    【参考资料】
    {context}
"""
    prompt_template = await get_prompt_template(db, "rag_system_prompt", default_system_template)
    system_prompt = prompt_template.format(context=context)
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})
    
    # 3. Call LLM
    if request.stream:
        async def stream_wrapper():
            collected_content = ""
            try:
                stream = await llm_service.chat_completion(messages, model=model_name, stream=True)
                async for chunk in stream:
                    collected_content += chunk
                    yield chunk
                if reference_links:
                    has_reference_section = ("引用文件" in collected_content or "📎" in collected_content)
                    if not has_reference_section:
                        yield reference_links
                        collected_content += reference_links
            except Exception as e:
                error_msg = f"\n\n❌ 错误: {str(e)}"
                yield error_msg
                collected_content = error_msg
            finally:
                try:
                    query_duration = time.time() - query_start_time
                    chat_log = ChatQueryLog(
                        username=username,
                        query_content=request.message,
                        initial_rag_results=initial_results,
                        reranked_results=reranked_results,
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
            if reference_links:
                has_reference_section = ("引用文件" in response_content or "📎" in response_content)
                if not has_reference_section:
                    response_content += reference_links
            
            query_duration = time.time() - query_start_time
            chat_log = ChatQueryLog(
                username=username,
                query_content=request.message,
                initial_rag_results=initial_results,
                reranked_results=reranked_results,
                llm_response=response_content,
                model_name=model_name,
                query_duration_seconds=query_duration
            )
            db.add(chat_log)
            await db.commit()
            return {"content": response_content}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM调用失败: {str(e)}")

@router.get("/chat-logs", response_model=PaginatedChatLogs)
async def get_chat_logs(
    page: int = 1,
    page_size: int = 15,
    username: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    query = select(ChatQueryLog)
    if username:
        query = query.where(ChatQueryLog.username == username)
    query = query.order_by(desc(ChatQueryLog.query_time))
    
    count_query = select(func.count()).select_from(ChatQueryLog)
    if username:
        count_query = count_query.where(ChatQueryLog.username == username)
    
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    total_pages = (total + page_size - 1) // page_size
    offset = (page - 1) * page_size
    
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    logs = result.scalars().all()
    
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
