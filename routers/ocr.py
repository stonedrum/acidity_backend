import logging
import os
import uuid
import shutil
import zipfile
import io
import httpx
import re
from typing import Optional, List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
from ..database import get_db
from ..models import OcrDocument, get_beijing_time, Clause, Document
from ..schemas import OcrDocumentOut, PaginatedOcrDocuments
from ..auth import get_current_user, check_role
from ..services.oss_service import oss_service
from ..services.mineru_service import mineru_service
from ..services.rag_service import rag_service
from ..services.ocr_process_service import process_mineru_result

router = APIRouter(tags=["OCR 解析管理"])
logger = logging.getLogger("ocr_router")

@router.post("/ocr/upload", response_model=OcrDocumentOut)
async def upload_for_ocr(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin", "editor"]))
):
    """上传文件用于 OCR 解析"""
    # 1. 保存临时文件并上传 OSS
    temp_dir = "temp_ocr"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. 上传到 OSS
        file_ext = os.path.splitext(file.filename)[1]
        oss_filename = f"{uuid.uuid4().hex}{file_ext}"
        with open(temp_file_path, "rb") as f:
            oss_key = oss_service.upload_file(f.read(), oss_filename, directory="ocr_raw")
        
        file_url = oss_service.get_file_url(oss_key)
        
        # 3. 写入数据库记录（初始状态：待提交）
        ocr_doc = OcrDocument(
            filename=file.filename, # 记录原始文件名
            original_file_url=file_url,
            task_id=None,
            ocr_status="待提交",
            rag_status="未提交",
            uploader=current_user["username"]
        )
        db.add(ocr_doc)
        await db.commit()
        await db.refresh(ocr_doc)

        # 4. 尝试提交任务到 MinerU
        try:
            task_id = await mineru_service.submit_task(file_url)
            ocr_doc.task_id = task_id
            ocr_doc.ocr_status = "待识别"
            await db.commit()
            await db.refresh(ocr_doc)
        except Exception as e:
            # 提交失败保持“待提交”状态，由后台任务重试
            print(f"[OCR Upload] Initial MinerU submission failed for {ocr_doc.id}: {e}")
        
        return ocr_doc
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/ocr/tasks/{ocr_id}/check", response_model=OcrDocumentOut)
async def check_ocr_task(
    ocr_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin", "editor"]))
):
    """手动触发检查特定 OCR 任务状态"""
    stmt = select(OcrDocument).where(OcrDocument.id == ocr_id)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # 如果是待提交，尝试提交
    if task.ocr_status == "待提交":
        try:
            task_id = await mineru_service.submit_task(task.original_file_url)
            task.task_id = task_id
            task.ocr_status = "待识别"
            await db.commit()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"提交 MinerU 失败: {str(e)}")
    
    # 如果是待识别，检查状态
    if task.ocr_status == "待识别" and task.task_id:
        try:
            logger.info(f"[Manual Check] Checking status for MinerU Task: {task.task_id}")
            status, zip_url = await mineru_service.get_task_status(task.task_id)
            logger.info(f"[Manual Check] MinerU Task {task.task_id} status: {status}")
            
            if status == "success" and zip_url:
                # 使用共享的服务处理结果
                success = await process_mineru_result(task, zip_url, db)
                if not success:
                    raise HTTPException(status_code=500, detail="处理 OCR 结果失败")
            elif status == "failed":
                logger.warning(f"[Manual Check] MinerU Task {task.task_id} failed.")
                task.ocr_status = "识别失败"
                await db.commit()
            else:
                logger.info(f"[Manual Check] MinerU Task {task.task_id} is still in progress: {status}")
        except Exception as e:
            logger.error(f"[Manual Check] Error processing task {task.task_id}: {e}")
            if not isinstance(e, HTTPException):
                raise HTTPException(status_code=500, detail=f"检查 MinerU 状态失败: {str(e)}")
            else:
                raise
            
    await db.refresh(task)
    return task

@router.get("/ocr/tasks", response_model=PaginatedOcrDocuments)
async def list_ocr_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1),
    ocr_status: Optional[str] = None,
    uploader: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin", "editor"]))
):
    """获取 OCR 解析任务列表"""
    stmt = select(OcrDocument)
    
    if current_user["role"] == "editor":
        stmt = stmt.where(OcrDocument.uploader == current_user["username"])
    elif uploader:
        stmt = stmt.where(OcrDocument.uploader == uploader)
    
    if ocr_status:
        stmt = stmt.where(OcrDocument.ocr_status == ocr_status)
    
    # 计算总数
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_res = await db.execute(count_stmt)
    total = total_res.scalar()
    
    # 排序并分页
    stmt = stmt.order_by(OcrDocument.upload_time.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(stmt)
    tasks = result.scalars().all()
    
    return PaginatedOcrDocuments(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
        items=tasks
    )

@router.delete("/ocr/tasks/{task_id}")
async def delete_ocr_task(
    task_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin", "editor"]))
):
    """删除 OCR 解析任务"""
    stmt = select(OcrDocument).where(OcrDocument.id == task_id)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if current_user["role"] == "editor" and task.uploader != current_user["username"]:
        raise HTTPException(status_code=403, detail="You can only delete your own tasks")
    
    await db.delete(task)
    await db.commit()
    return {"message": "OCR task deleted"}

@router.post("/ocr/tasks/{ocr_id}/submit_rag", response_model=OcrDocumentOut)
async def submit_ocr_to_rag(
    ocr_id: uuid.UUID,
    kb_type: str = Form(...),
    region_level: Optional[str] = Form(None),
    province: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin", "admin", "editor"]))
):
    """将 OCR 解析结果（ZIP 中的 Markdown）提交到 RAG 数据库"""
    # 1. 获取 OCR 任务信息
    stmt = select(OcrDocument).where(OcrDocument.id == ocr_id)
    result = await db.execute(stmt)
    ocr_doc = result.scalar_one_or_none()
    
    if not ocr_doc:
        raise HTTPException(status_code=404, detail="OCR 任务未找到")
    if ocr_doc.ocr_status != "已识别":
        raise HTTPException(status_code=400, detail="OCR 任务尚未完成识别")
    if not ocr_doc.result_file_url:
        raise HTTPException(status_code=400, detail="OCR 任务没有结果文件")
    if ocr_doc.rag_status == "已提交":
        raise HTTPException(status_code=400, detail="任务已提交过 RAG 数据库")

    try:
        # 2. 下载并解压 ZIP
        async with httpx.AsyncClient() as client:
            resp = await client.get(ocr_doc.result_file_url, timeout=60.0)
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail="无法下载 OCR 结果文件")
            
        zip_data = io.BytesIO(resp.content)
        with zipfile.ZipFile(zip_data) as zf:
            # MinerU 结果通常包含一个 .md 文件
            md_files = [n for n in zf.namelist() if n.endswith('.md')]
            if not md_files:
                raise HTTPException(status_code=400, detail="ZIP 中未找到 Markdown 内容")
            
            # 优先找 content.md
            md_file_name = "content.md" if "content.md" in md_files else md_files[0]
            with zf.open(md_file_name) as f:
                md_content = f.read().decode('utf-8')

        # 3. 创建关联文档记录
        # 提取文件名
        original_filename = os.path.basename(ocr_doc.original_file_url.split('?')[0])
        doc_filename = f"OCR_{ocr_id.hex[:6]}_{original_filename}"
        
        # 用结果 ZIP 的 key 也可以，或者根据需要调整
        result_key = ocr_doc.result_file_url.split('/')[-1].split('?')[0]
        
        doc = Document(
            filename=doc_filename,
            oss_key=f"ocr_results/{result_key}",
            uploader=ocr_doc.uploader,
            kb_type=kb_type,
            region_level=region_level,
            province=province,
            city=city
        )
        db.add(doc)
        await db.flush() # 获取 doc.id

        # 4. 解析并插入 Clauses
        # 按标题分割: ## 标题
        sections = re.split(r'\n(?=##?\s)', "\n" + md_content)
        inserted_count = 0
        for section in sections:
            section = section.strip()
            if not section: continue
            
            lines = section.split('\n')
            title = lines[0].replace('#', '').strip() if lines[0].startswith('#') else "OCR 条目"
            body = '\n'.join(lines[1:]).strip() if lines[0].startswith('#') else section
            
            if not body: continue
            
            # 超过 512 字，按段落/文字长度拆分并组合
            chunks = []
            paragraphs = re.split(r'\n\s*\n', body)
            current_chunk = ""
            for p in paragraphs:
                p = p.strip()
                if not p: continue
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

            for chunk_text in chunks:
                embedding = rag_service.get_embedding(chunk_text)
                clause = Clause(
                    doc_id=doc.id,
                    kb_type=kb_type,
                    chapter_path=title,
                    content=chunk_text,
                    creator=current_user["username"],
                    import_method="ocr 导入",
                    embedding=embedding,
                    is_verified=True, # OCR 手工确认提交的设为已校验
                    region_level=region_level,
                    province=province,
                    city=city
                )
                db.add(clause)
                inserted_count += 1

        # 5. 更新任务状态
        ocr_doc.rag_status = "已提交"
        ocr_doc.submit_time = get_beijing_time()
        ocr_doc.submitter = current_user["username"]
        
        await db.commit()
        await db.refresh(ocr_doc)
        return ocr_doc
        
    except Exception as e:
        await db.rollback()
        print(f"[submit_ocr_to_rag] Error: {e}")
        raise HTTPException(status_code=500, detail=f"提交 RAG 失败: {str(e)}")
