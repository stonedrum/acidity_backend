from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from ..database import get_db
from ..models import Clause, Document
from ..schemas import (
    ClauseOut, ClauseCreate, ClauseUpdate, PaginatedClauses, 
    ClauseBatchCreate, BatchInsertResult, ClauseBatchUpdate
)
from ..auth import get_current_user, check_role
from ..services.rag_service import rag_service

router = APIRouter(tags=["条款管理"])

@router.get("/clauses", response_model=PaginatedClauses)
async def list_clauses(
    page: int = 1,
    page_size: int = 15,
    kb_type: Optional[str] = None,
    doc_id: Optional[UUID] = None,
    keyword: Optional[str] = None,
    is_verified: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin"]))
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
    
    # 分页排序
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
                page_number=c.page_number,
                is_verified=c.is_verified,
                doc_id=c.doc_id,
                doc_name=doc_map.get(c.doc_id, "手动新增")
            ) for c in clauses
        ]
    )

@router.post("/clauses", response_model=ClauseOut)
async def create_clause(
    data: ClauseCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin"]))
):
    """手动创建知识条款"""
    embedding = rag_service.get_embedding(data.content)
    new_clause = Clause(
        kb_type=data.kb_type,
        chapter_path=data.chapter_path,
        content=data.content,
        page_number=data.page_number,
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
        page_number=new_clause.page_number,
        is_verified=new_clause.is_verified,
        doc_id=new_clause.doc_id,
        doc_name=doc_name
    )

@router.post("/clauses/batch", response_model=BatchInsertResult)
async def batch_create_clauses(
    data: ClauseBatchCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin"]))
):
    """批量导入知识条款"""
    if not data.items:
        raise HTTPException(status_code=400, detail="Items cannot be empty")

    # 调试：打印接收到的第一条数据
    print(f"[Batch Debug] First item received: {data.items[0].dict()}")

    new_clauses: List[Clause] = []
    print(f"[Batch Import] Received {len(data.items)} items")
    for i, item in enumerate(data.items):
        if not item.content:
            continue
        if i < 5:  # 只打印前5条用于调试
            print(f"[Batch Import] Item {i}: page_number={item.page_number}, chapter={item.chapter_path[:20]}")
        embedding = rag_service.get_embedding(item.content)
        new_clauses.append(
            Clause(
                kb_type=data.kb_type,
                chapter_path=item.chapter_path,
                content=item.content,
                page_number=item.page_number,
                doc_id=data.doc_id,
                embedding=embedding,
                is_verified=True
            )
        )

    if not new_clauses:
        raise HTTPException(status_code=400, detail="No valid items to insert")

    db.add_all(new_clauses)
    await db.commit()
    return BatchInsertResult(inserted=len(new_clauses))

@router.post("/clauses/batch-update")
async def batch_update_clauses(
    data: ClauseBatchUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin"]))
):
    """批量更新知识条款"""
    if not data.ids:
        raise HTTPException(status_code=400, detail="IDs cannot be empty")
    
    stmt = select(Clause).where(Clause.id.in_(data.ids))
    result = await db.execute(stmt)
    clauses = result.scalars().all()
    
    if not clauses:
        raise HTTPException(status_code=404, detail="No clauses found for given IDs")
    
    for clause in clauses:
        if data.kb_type is not None:
            clause.kb_type = data.kb_type
        if data.doc_id is not None:
            clause.doc_id = data.doc_id
        if data.is_verified is not None:
            clause.is_verified = data.is_verified
            
    await db.commit()
    return {"updated": len(clauses)}

@router.put("/clauses/{clause_id}", response_model=ClauseOut)
async def update_clause(
    clause_id: UUID,
    data: ClauseUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin"]))
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
    if data.page_number is not None:
        clause.page_number = data.page_number
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
        page_number=clause.page_number,
        is_verified=clause.is_verified,
        doc_id=clause.doc_id,
        doc_name=doc_name
    )

@router.delete("/clauses/{clause_id}")
async def delete_clause(
    clause_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin"]))
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
