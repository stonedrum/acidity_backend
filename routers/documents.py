import os
import shutil
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from ..database import get_db
from ..models import Document, Clause
from ..schemas import DocumentOut, DocumentUpdate, PaginatedDocuments
from ..auth import get_current_user, check_role
from ..services.oss_service import oss_service
from ..services.pdf_service import pdf_service
from ..services.rag_service import rag_service

router = APIRouter(tags=["文档管理"])

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...), 
    kb_type: str = Form(...),
    region_level: Optional[str] = Form(None),
    province: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db), 
    current_user: dict = Depends(check_role(["sysadmin", "admin", "editor"]))
):
    # ...
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
            uploader=current_user["username"],
            kb_type=kb_type,
            region_level=region_level,
            province=province,
            city=city
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
                page_number=item.get("page_number"),
                creator=current_user["username"],
                import_method="pdf 导入",
                embedding=embedding,
                is_verified=False, # PDF 自动导入的设为未校验
                region_level=region_level,
                province=province,
                city=city
            )
            db.add(clause)
        
        await db.commit()
        return {"message": "Upload and processing successful", "doc_id": doc.id}
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/documents")
async def create_document_simple(
    file: UploadFile = File(...), 
    kb_type: str = Form(...),
    auto_import: bool = Form(False),
    region_level: Optional[str] = Form(None),
    province: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db), 
    current_user: dict = Depends(check_role(["sysadmin", "admin", "editor"]))
):
    """上传文档并创建记录，可选是否进行 RAG 解析"""
    print(f"[Debug] create_document_simple received: region_level={region_level}, province={province}, city={city}")
    # 1. 检查文件名是否重复
    existing_doc = await db.execute(select(Document).where(Document.filename == file.filename))
    if existing_doc.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 已存在")

    # 2. 保存临时文件并上传 OSS
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        file_ext = os.path.splitext(file.filename)[1]
        oss_filename = f"{uuid.uuid4().hex}{file_ext}"
        with open(temp_file_path, "rb") as f:
            oss_key = oss_service.upload_file(f.read(), oss_filename, directory="laws")
        
        # 3. 创建数据库记录
        doc = Document(
            filename=file.filename, 
            oss_key=oss_key,
            uploader=current_user["username"],
            kb_type=kb_type,
            region_level=region_level,
            province=province,
            city=city
        )
        db.add(doc)
        await db.flush() # 获取 doc.id

        # 4. 如果开启自动导入
        inserted_count = 0
        if auto_import and file_ext.lower() == ".pdf":
            try:
                clauses_data = pdf_service.parse_pdf(temp_file_path)
                for item in clauses_data:
                    embedding = rag_service.get_embedding(item["content"])
                    clause = Clause(
                        doc_id=doc.id,
                        kb_type=kb_type,
                        chapter_path=item["chapter_path"],
                        content=item["content"],
                        page_number=item.get("page_number"),
                        creator=current_user["username"],
                        import_method="pdf 导入",
                        embedding=embedding,
                        is_verified=False, # PDF 自动导入的设为未校验
                        region_level=region_level,
                        province=province,
                        city=city
                    )
                    db.add(clause)
                    inserted_count += 1
            except Exception as e:
                print(f"[create_document_simple] 自动解析 PDF 失败: {e}")
                # 即使解析失败，文档还是传上去了，这里可以选择报错或者继续
        
        await db.commit()
        await db.refresh(doc)
        
        res = {
            "id": doc.id,
            "filename": doc.filename,
            "kb_type": doc.kb_type,
            "auto_import": auto_import,
            "inserted_clauses": inserted_count,
            "region_level": region_level,
            "province": province,
            "city": city
        }
        return res
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.put("/documents/{doc_id}", response_model=DocumentOut)
async def update_document(
    doc_id: uuid.UUID,
    data: DocumentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin", "editor"]))
):
    """更新文档信息"""
    stmt = select(Document).where(Document.id == doc_id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # 信息录入员只能编辑自己的文档
    if current_user["role"] == "editor" and doc.uploader != current_user["username"]:
        raise HTTPException(status_code=403, detail="You can only update your own documents")
    
    if data.filename is not None:
        # 检查新文件名是否与其他文档冲突
        if data.filename != doc.filename:
            existing = await db.execute(select(Document).where(Document.filename == data.filename))
            if existing.scalar_one_or_none():
                raise HTTPException(status_code=400, detail=f"文件名 '{data.filename}' 已被使用")
        doc.filename = data.filename
        
    if data.kb_type is not None:
        # 如果修改了文档的 kb_type，通常也需要同步修改关联的 clauses 的 kb_type
        doc.kb_type = data.kb_type
        from sqlalchemy import update
        await db.execute(
            update(Clause).where(Clause.doc_id == doc_id).values(kb_type=data.kb_type)
        )
        
    if data.region_level is not None or data.province is not None or data.city is not None:
        if data.region_level is not None: doc.region_level = data.region_level
        if data.province is not None: doc.province = data.province
        if data.city is not None: doc.city = data.city
        
        # 同步更新关联的条款
        from sqlalchemy import update
        await db.execute(
            update(Clause).where(Clause.doc_id == doc_id).values(
                region_level=doc.region_level,
                province=doc.province,
                city=doc.city
            )
        )
        
    await db.commit()
    await db.refresh(doc)
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        uploader=doc.uploader,
        kb_type=doc.kb_type,
        upload_time=doc.upload_time,
        file_url=oss_service.get_file_url(doc.oss_key) if doc.oss_key else None,
        region_level=doc.region_level,
        province=doc.province,
        city=doc.city
    )

@router.get("/documents", response_model=PaginatedDocuments)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    kb_type: Optional[str] = None,
    keyword: Optional[str] = None,
    uploader: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin", "editor"]))
):
    """获取所有文档列表（支持模糊匹配文件名、按类型筛选、分页）"""
    stmt = select(Document)
    
    # 信息录入员只看自己的
    if current_user["role"] == "editor":
        stmt = stmt.where(Document.uploader == current_user["username"])
    elif uploader:
        stmt = stmt.where(Document.uploader == uploader)
        
    if kb_type:
        stmt = stmt.where(Document.kb_type == kb_type)
    if keyword:
        stmt = stmt.where(Document.filename.ilike(f"%{keyword}%"))
    
    # 计算总数
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_res = await db.execute(count_stmt)
    total = total_res.scalar()
    
    # 分页排序
    stmt = stmt.order_by(Document.upload_time.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(stmt)
    docs = result.scalars().all()
    
    # 填充结果
    items = [
        DocumentOut(
            id=d.id,
            filename=d.filename,
            uploader=d.uploader,
            kb_type=d.kb_type,
            upload_time=d.upload_time,
            file_url=oss_service.get_file_url(d.oss_key) if d.oss_key else None,
            region_level=d.region_level,
            province=d.province,
            city=d.city
        ) for d in docs
    ]
    
    return PaginatedDocuments(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
        items=items
    )

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin", "editor"]))
):
    """删除文档及其关联的所有条款"""
    stmt = select(Document).where(Document.id == doc_id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # 信息录入员只能删除自己的
    if current_user["role"] == "editor" and doc.uploader != current_user["username"]:
        raise HTTPException(status_code=403, detail="You can only delete your own documents")
    
    # 删除 OSS 文件（可选，如果需要同步删除）
    # oss_service.delete_file(doc.oss_key)
    
    await db.delete(doc)
    await db.commit()
    return {"message": "Document and associated clauses deleted"}

@router.post("/documents/{doc_id}/import-markdown")
async def import_markdown(
    doc_id: uuid.UUID,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin", "editor"]))
):
    """从 Markdown 文件导入知识条目，按二级标题拆分并处理超长内容"""
    # 1. 获取文档信息
    stmt = select(Document).where(Document.id == doc_id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 信息录入员只能导入到自己的文档
    if current_user["role"] == "editor" and doc.uploader != current_user["username"]:
        raise HTTPException(status_code=403, detail="You can only import to your own documents")

    # 2. 读取并解析 Markdown
    content = await file.read()
    text = content.decode("utf-8")
    
    # 按二级标题拆分: ## 标题
    import re
    sections = re.split(r'\n(?=##\s)', "\n" + text)
    
    inserted_count = 0
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # 提取标题和内容
        lines = section.split('\n')
        title = lines[0].replace('##', '').strip() if lines[0].startswith('##') else "未命名章节"
        body = '\n'.join(lines[1:]).strip() if lines[0].startswith('##') else section
        
        if not body:
            continue

        # 拆分逻辑
        chunks = []
        if len(body) <= 1024 and '|' not in body: # 如果内容不多且没表格，直接存
            chunks.append(body)
        else:
            # 超过 1024 字，或者包含表格，按段落/表格拆分并组合
            paragraphs = re.split(r'\n\s*\n', body)
            current_chunk = ""
            
            def is_table(p_text):
                # 简单的表格判定：包含 | 且有类似 |---| 的分隔行
                lines = p_text.strip().split('\n')
                if len(lines) < 2:
                    return False
                has_pipe = any('|' in line for line in lines)
                has_sep = any(re.match(r'^\s*\|?[:\s-]*\|[:\s-]*\|?[:\s-]*', line) for line in lines)
                return has_pipe and has_sep

            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue
                
                # 如果是表格，必须作为一个整体条目
                if is_table(p):
                    # 先结算之前的 chunk
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    # 将表格独立作为一个 chunk
                    chunks.append(p)
                    continue

                # 非表格按 512 字逻辑合并
                if len(current_chunk) + len(p) > 512 and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = p
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + p
                    else:
                        current_chunk = p
            
            if current_chunk:
                chunks.append(current_chunk.strip())

        # 3. 向量化并插入数据库
        for chunk_text in chunks:
            embedding = rag_service.get_embedding(chunk_text)
            clause = Clause(
                doc_id=doc.id,
                kb_type=doc.kb_type,
                chapter_path=title,
                content=chunk_text,
                creator=current_user["username"],
                import_method="markdown 录入",
                embedding=embedding,
                is_verified=False, # Markdown 导入的设为未校验
                region_level=doc.region_level,
                province=doc.province,
                city=doc.city
            )
            db.add(clause)
            inserted_count += 1
            
    await db.commit()
    return {"status": "ok", "inserted": inserted_count}
