import logging
import time
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from ..database import get_db
from ..models import Document, Clause, ChatQueryLog, ModelComparisonVote, DictType, DictData
from ..schemas import (
    SearchQuery, ClauseOut, ChatRequest, ChatQueryLogOut, PaginatedChatLogs,
    ComparisonVoteCreate, ComparisonVoteOut, PaginatedComparisonVotes, ComparisonStats
)
from ..auth import get_current_user, check_role
from ..services.rag_service import rag_service
from ..services.llm_service import llm_service
from ..services.oss_service import oss_service
from ..services.prompt_service import get_prompt_template
from ..config import settings

router = APIRouter(tags=["问答检索"])
logger = logging.getLogger("chat")

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
    current_user: dict = Depends(get_current_user)
):
    query_start_time = time.time()
    username = current_user["username"]
    # 不再在这里强行回退到 settings.LLM_MODEL，交给 llm_service 动态判断
    model_name = request.model
    
    # 1. RAG
    effective_kb_type = request.kb_type
    intent_info = None
    
    # 如果用户没有指定知识库类型，进行意图识别
    if not effective_kb_type:
        try:
            # 获取所有可用的知识库类型
            stmt = (
                select(DictData.value, DictData.label)
                .join(DictType)
                .where(DictType.type_name == "kb_type", DictData.is_active == True)
            )
            dict_res = await db.execute(stmt)
            kb_types = dict_res.all()
            
            if kb_types:
                type_desc = "\n".join([f"- {t.value} ({t.label})" for t in kb_types])
                intent_prompt = f"""你是一个意图识别专家。请根据用户的问题，将其归类到最相关的知识库类型中。
当前的知识库类型列表如下：
{type_desc}

用户问题：{request.message}

任务：
1. 从上述列表中选择一个最匹配的“知识库类型值”（即括号前的英文/标识符部分）。
2. 如果没有任何类型匹配，请回复 "None"。
3. 你的回复必须【仅包含】选中的类型值或 "None"，不要有任何其他文字。"""
                
                # 调用大模型进行意图识别（使用默认模型，非流式）
                detected_type_raw = await llm_service.chat_completion(
                    [{"role": "user", "content": intent_prompt}],
                    model=None, # 使用系统默认模型
                    stream=False
                )
                detected_type = detected_type_raw.strip()
                
                intent_info = {
                    "prompt": intent_prompt,
                    "response": detected_type_raw,
                    "detected_type": detected_type
                }
                
                # 验证识别出的类型是否在有效列表中
                valid_values = [t.value for t in kb_types]
                if detected_type in valid_values:
                    effective_kb_type = detected_type
                    logger.info(f"[Intent] 自动识别知识库类型: {effective_kb_type}")
        except Exception as e:
            logger.warning(f"[Intent Error] 意图识别失败: {e}")
            intent_info = {"error": str(e)}

    # 查询 RAG：首次用当前问题；若重排后无结果则依次拼接 1～4 条历史用户问句再查，每次都以「重排后是否有结果」为准
    history_list = request.history if request.history is not None else []
    user_messages = [m.content.strip() for m in history_list if getattr(m, "role", "") == "user"]
    logger.info(f"[RAG] 收到历史条数={len(history_list)}, 其中用户问句数={len(user_messages)}, 当前消息={request.message.strip()[:60]}...")
    initial_results, reranked_results = None, None
    rag_query_steps = []  # 记录每次尝试的拼接查询过程，供日志详情展示
    for prev_count in range(0, 5):
        if prev_count == 0:
            rag_query = request.message.strip()
            logger.info(f"[RAG] 第 1 次尝试：仅当前问句，query_len={len(rag_query)}")
        else:
            if len(user_messages) < prev_count:
                logger.info(f"[RAG] 历史用户问句不足 {prev_count} 条，停止拼接尝试")
                break
            rag_query = " ".join(user_messages[-prev_count:] + [request.message.strip()])
            logger.info(f"[RAG] 第 {prev_count + 1} 次尝试：拼接 {prev_count} 条历史，query_len={len(rag_query)}")
        initial_results, reranked_results = await rag_service.search_and_rerank(
            rag_query,
            db,
            kb_type=effective_kb_type,
            return_initial_results=True
        )
        n_rerank = len(reranked_results) if reranked_results else 0
        logger.info(f"[RAG] 本步重排结果数={n_rerank}, had_rerank_results={bool(reranked_results)}")
        # 记录本步：用于查询日志详情页展示拼接过程
        rag_query_steps.append({
            "step": len(rag_query_steps) + 1,
            "history_count": prev_count,
            "query_used": rag_query[:500] + ("..." if len(rag_query) > 500 else ""),
            "had_rerank_results": bool(reranked_results),
        })
        if reranked_results:
            if prev_count > 0:
                logger.info(f"[RAG] 拼接 {prev_count} 条历史后命中重排结果，停止尝试")
            break
        if prev_count > 0:
            logger.info(f"[RAG] 已拼接 {prev_count} 条历史仍无重排结果，继续尝试更多历史")

    # 最多 5 轮尝试后仍无匹配参考资料，提示用户给出完整描述
    if not reranked_results:
        no_result_msg = "未找到相关参考资料，请您给出更完整的问题描述后再试。"
        
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
                intent_info=intent_info, # 记录意图识别详情
                rag_query_steps=rag_query_steps,
                model_name=model_name,
                query_duration_seconds=query_duration
            )
            db.add(chat_log)
            await db.commit()
        except Exception as e:
            logger.error(f"[ERROR] 保存查询记录失败: {e}")

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
                page_info = f" 页码：{clause.page_number}" if clause.page_number else ""
                context += f"【参考资料{i+1}】(来自文档：{doc_name}{page_info}) 章节路径：{clause.chapter_path}\n内容：{clause.content}\n\n"
                if clause.doc_id:
                    referenced_doc_ids.add(clause.doc_id)
                # 将页码存入有效条目中（用于生成下方引用链接）
                item["page_number"] = clause.page_number
                valid_reranked_items.append(item)
        
        # 更新重排结果为实际查到的有效条目（记录日志用）
        reranked_results = valid_reranked_items

    # 构建去重后的引用链接（考虑页码）
    reference_links = ""
    if valid_reranked_items:
        # 获取所有文档详情
        doc_ids = set(UUID(item["doc_id"]) if isinstance(item["doc_id"], str) else item["doc_id"] 
                     for item in valid_reranked_items if item.get("doc_id"))
        doc_details = {}
        if doc_ids:
            doc_stmt = select(Document).where(Document.id.in_(doc_ids))
            doc_result = await db.execute(doc_stmt)
            for d in doc_result.scalars().all():
                if d.oss_key:
                    doc_details[str(d.id)] = d

        # 构建唯一的 (doc_id, page_number) 集合
        unique_refs = set()
        for item in valid_reranked_items:
            did = str(item.get("doc_id"))
            if did in doc_details:
                unique_refs.add((did, item.get("page_number")))

        if unique_refs:
            # 排序：先按文件名，再按页码
            sorted_refs = sorted(list(unique_refs), key=lambda x: (doc_details[x[0]].filename, x[1] or 0))
            
            reference_links = "\n\n---\n**📎 引用文件：**\n"
            for did, pnum in sorted_refs:
                doc = doc_details[did]
                file_url = oss_service.get_file_url(doc.oss_key)
                page_suffix = ""
                display_name = doc.filename
                
                if pnum:
                    page_suffix = f" 第 {pnum} 页"
                    display_name += f" (第 {pnum} 页)"
                    # 如果是 PDF，添加 #page=N 参数
                    if doc.filename.lower().endswith(".pdf"):
                        # 处理 URL 可能已经包含参数的情况
                        connector = "&" if "?" in file_url else "?"
                        # 但实际上 #page=N 应该是在最后
                        file_url += f"#page={pnum}"
                
                reference_links += f"- [{display_name}]({file_url})\n"
    
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
    
    # 仅将当前问题发给 LLM；历史对话只用于 RAG 补充查询，不传入 LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.message}
    ]
    
    # 3. Call LLM
    if request.stream:
        async def stream_wrapper():
            collected_content = ""
            try:
                # 使用统一的方法获取实际模型名
                actual_model_name, _ = llm_service.get_actual_model_info(model_name)

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
                        intent_info=intent_info, # 记录意图识别详情
                        rag_query_steps=rag_query_steps,
                        model_name=actual_model_name, # 使用解析后的实际模型名
                        query_duration_seconds=query_duration
                    )
                    db.add(chat_log)
                    await db.commit()
                except Exception as e:
                    logger.error(f"[ERROR] 保存查询记录失败: {e}")
        return StreamingResponse(stream_wrapper(), media_type="text/plain; charset=utf-8")
    else:
        try:
            # 使用统一的方法获取实际模型名
            actual_model_name, _ = llm_service.get_actual_model_info(model_name)

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
                intent_info=intent_info, # 记录意图识别详情
                rag_query_steps=rag_query_steps,
                model_name=actual_model_name, # 使用解析后的实际模型名
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
    current_user: dict = Depends(get_current_user)
):
    # 如果是普通用户，只能看自己的日志
    final_username = username
    if current_user["role"] == "user":
        final_username = current_user["username"]
    
    query = select(ChatQueryLog)
    if final_username:
        query = query.where(ChatQueryLog.username == final_username)
    query = query.order_by(desc(ChatQueryLog.query_time))
    
    count_query = select(func.count()).select_from(ChatQueryLog)
    if final_username:
        count_query = count_query.where(ChatQueryLog.username == final_username)
    
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
            llm_messages=log.llm_messages,
            intent_info=log.intent_info,
            rag_query_steps=log.rag_query_steps,
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
    current_user: dict = Depends(get_current_user)
):
    # 尝试从最近的查询日志中恢复 llm_messages (如果前端没传)
    final_messages = vote.llm_messages
    if not final_messages:
        # 查找该用户最近一次针对该内容的查询日志
        stmt = select(ChatQueryLog.llm_messages).where(
            ChatQueryLog.username == current_user["username"],
            ChatQueryLog.query_content == vote.query_content
        ).order_by(desc(ChatQueryLog.query_time)).limit(1)
        res = await db.execute(stmt)
        final_messages = res.scalar()

    new_vote = ModelComparisonVote(
        username=current_user["username"],
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
    current_user: dict = Depends(check_role(["sysadmin", "admin"]))
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
    current_user: dict = Depends(check_role(["sysadmin", "admin"]))
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
