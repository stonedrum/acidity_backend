import os
import shutil
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db
from ..models import Document, Clause
from ..schemas import DocumentOut
from ..auth import get_current_user
from ..services.oss_service import oss_service
from ..services.pdf_service import pdf_service
from ..services.rag_service import rag_service

router = APIRouter(tags=["文档管理"])

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...), 
    kb_type: str = Form(...),
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
        # 2. Generate UUID-based filename
        file_ext = os.path.splitext(file.filename)[1]
        uuid_name = str(uuid.uuid4()).replace('-', '')
        oss_filename = f"{uuid_name}{file_ext}"
        
        # 3. Upload to OSS
        with open(temp_file_path, "rb") as f:
            oss_key = oss_service.upload_file(f.read(), oss_filename, directory="laws")
        
        # 4. Check if filename already exists
        existing_doc = await db.execute(select(Document).where(Document.filename == file.filename))
        if existing_doc.scalar_one_or_none():
            raise HTTPException(
                status_code=400, 
                detail=f"文件 '{file.filename}' 已存在，请使用不同的文件名"
            )
        
        # 5. Create Document record
        doc = Document(
            filename=file.filename, 
            oss_key=oss_key,
            uploader=current_user,
            kb_type=kb_type
        )
        db.add(doc)
        await db.flush()
        
        # 6. Parse PDF
        clauses_data = pdf_service.parse_pdf(temp_file_path)
        
        # 7. Embedding and Save Clauses
        for item in clauses_data:
            embedding = rag_service.get_embedding(item["content"])
            clause = Clause(
                doc_id=doc.id,
                kb_type=kb_type,
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

@router.get("/documents", response_model=List[DocumentOut])
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
