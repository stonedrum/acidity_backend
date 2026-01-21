import time
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from ..database import get_db
from ..models import Document, Clause, ChatQueryLog, ModelComparisonVote
from ..schemas import (
    SearchQuery, ClauseOut, ChatRequest, ChatQueryLogOut, PaginatedChatLogs,
    ComparisonVoteCreate, ComparisonVoteOut, PaginatedComparisonVotes, ComparisonStats
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
    
    # 如果没有找到任何匹配的参考资料，直接返回提示信息，不再调用大模型
    if not reranked_results:
        no_result_msg = "您好，未找到相关参考资料。"
        
        # 记录日志
        try:
            query_duration = time.time() - query_start_time
            chat_log = ChatQueryLog(
                username=username,
                query_content=request.message,
                initial_rag_results=initial_results,
                reranked_results=[],
                llm_response=no_result_msg,
                llm_messages=[{"role": "user", "content": request.message}], # 仅记录当前提问
                model_name=model_name,
                query_duration_seconds=query_duration
            )
            db.add(chat_log)
            await db.commit()
        except Exception as e:
            print(f"[ERROR] 保存查询记录失败: {e}")

        if request.stream:
            async def empty_stream():
                yield no_result_msg
            return StreamingResponse(empty_stream(), media_type="text/plain; charset=utf-8")
        else:
            return {"content": no_result_msg}

    context = ""
    referenced_doc_ids = set()
    
    # 优化：批量获取完整的条款和对应的文档信息
    if reranked_results:
        clause_ids = [UUID(item["clause_id"]) if isinstance(item["clause_id"], str) else item["clause_id"] for item in reranked_results]
        # 使用 join 预加载文档信息
        from sqlalchemy.orm import joinedload
        stmt = select(Clause).options(joinedload(Clause.document)).where(Clause.id.in_(clause_ids))
        result = await db.execute(stmt)
        # 将结果转为字典方便按顺序查找
        clauses_map = {c.id: c for c in result.scalars().all()}
        
        valid_reranked_items = []
        for i, item in enumerate(reranked_results):
            cid = UUID(item["clause_id"]) if isinstance(item["clause_id"], str) else item["clause_id"]
            clause = clauses_map.get(cid)
            if clause:
                doc_name = clause.document.filename if clause.document else "手动录入"
                context += f"【参考资料{i+1}】(来自文档：{doc_name}) 章节路径：{clause.chapter_path}\n内容：{clause.content}\n\n"
                if clause.doc_id:
                    referenced_doc_ids.add(clause.doc_id)
                valid_reranked_items.append(item)
        
        # 更新重排结果为实际查到的有效条目（记录日志用）
        reranked_results = valid_reranked_items

    referenced_docs = []
    if referenced_doc_ids:
        # 移除可能存在的 None
        clean_doc_ids = [rid for rid in referenced_doc_ids if rid is not None]
        if clean_doc_ids:
            stmt = select(Document).where(Document.id.in_(clean_doc_ids))
            result = await db.execute(stmt)
            referenced_docs = result.scalars().all()
    
    reference_links = ""
    if referenced_docs:
        # 过滤掉没有 oss_key 的文档（手动新增的文档可能没上传文件）
        valid_docs = [d for d in referenced_docs if d.oss_key]
        if valid_docs:
            reference_links = "\n\n---\n**📎 引用文件：**\n"
            for doc in valid_docs:
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
                        llm_messages=messages,  # 保存完整的消息列表
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
                llm_messages=messages,  # 保存完整的消息列表
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
            llm_messages=log.llm_messages, # 新增字段
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

@router.post("/comparison/vote")
async def save_comparison_vote(
    vote: ComparisonVoteCreate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    # 尝试从最近的查询日志中恢复 llm_messages (如果前端没传)
    final_messages = vote.llm_messages
    if not final_messages:
        # 查找该用户最近一次针对该内容的查询日志
        stmt = select(ChatQueryLog.llm_messages).where(
            ChatQueryLog.username == (current_user or "anonymous"),
            ChatQueryLog.query_content == vote.query_content
        ).order_by(desc(ChatQueryLog.query_time)).limit(1)
        res = await db.execute(stmt)
        final_messages = res.scalar()

    new_vote = ModelComparisonVote(
        username=current_user or "anonymous",
        query_content=vote.query_content,
        qwen_response=vote.qwen_response,
        deepseek_response=vote.deepseek_response,
        winner=vote.winner,
        llm_messages=final_messages
    )
    db.add(new_vote)
    await db.commit()
    return {"status": "ok", "message": "投票已记录"}

@router.get("/comparison/votes", response_model=PaginatedComparisonVotes)
async def get_comparison_votes(
    page: int = 1,
    page_size: int = 15,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    query = select(ModelComparisonVote).order_by(desc(ModelComparisonVote.vote_time))
    
    count_query = select(func.count()).select_from(ModelComparisonVote)
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    votes = result.scalars().all()
    
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    
    return PaginatedComparisonVotes(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        items=votes
    )

@router.get("/comparison/stats", response_model=ComparisonStats)
async def get_comparison_stats(
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    # 总票数
    total_stmt = select(func.count()).select_from(ModelComparisonVote)
    total_res = await db.execute(total_stmt)
    total = total_res.scalar() or 0
    
    # Qwen 胜场 (winner=1)
    qwen_stmt = select(func.count()).select_from(ModelComparisonVote).where(ModelComparisonVote.winner == 1)
    qwen_res = await db.execute(qwen_stmt)
    qwen_wins = qwen_res.scalar() or 0
    
    # DeepSeek 胜场 (winner=2)
    ds_stmt = select(func.count()).select_from(ModelComparisonVote).where(ModelComparisonVote.winner == 2)
    ds_res = await db.execute(ds_stmt)
    ds_wins = ds_res.scalar() or 0
    
    return ComparisonStats(
        total_votes=total,
        qwen_wins=qwen_wins,
        deepseek_wins=ds_wins
    )
