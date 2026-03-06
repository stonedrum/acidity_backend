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
from ..models import OcrDocument, get_beijing_time, Clause, Document, SystemConfig
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
        
        # 编码文件名以避免 HTTP Header 中的中文乱码
        from urllib.parse import quote
        encoded_filename = quote(file.filename)
        headers = {
            'Content-Disposition': f'inline; filename="{encoded_filename}"; filename*=UTF-8\'\'{encoded_filename}'
        }
        
        with open(temp_file_path, "rb") as f:
            oss_key = oss_service.upload_file(f.read(), oss_filename, directory="ocr_raw", headers=headers)
        
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
    rag_status: Optional[str] = None,
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
    
    if rag_status:
        stmt = stmt.where(OcrDocument.rag_status == rag_status)
    
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
    
    # 确定要删除的 OSS 文件列表
    files_to_delete = []
    
    # 1. 检查原始 PDF 文件是否正在被其他模块（如文档管理）使用
    if task.original_file_url:
        # 提取 OSS key
        from urllib.parse import urlparse
        parsed_url = urlparse(task.original_file_url)
        oss_key = parsed_url.path.lstrip('/')
        
        # 检查 documents 表中是否有记录引用了此文件
        doc_stmt = select(Document).where(Document.oss_key == oss_key)
        doc_res = await db.execute(doc_stmt)
        if not doc_res.scalar_one_or_none():
            # 没有被文档管理引用，可以安全删除
            files_to_delete.append(task.original_file_url)
        else:
            logger.info(f"Original PDF file {oss_key} is in use by Document Management, skipping deletion.")
            
    # 2. 其它 OCR 特有的结果文件（ZIP, JSON, MD）总是可以删除，因为它们只属于这个 OCR 记录
    if task.result_file_url:
        files_to_delete.append(task.result_file_url)
    if task.json_file_url:
        files_to_delete.append(task.json_file_url)
    if task.md_file_url:
        files_to_delete.append(task.md_file_url)

    # 执行删除
    for url in files_to_delete:
        oss_service.delete_file(url)
            
    await db.delete(task)
    await db.commit()
    return {"message": "OCR task deleted, associated OSS files cleaned up (if not in use)"}

def split_sentences(text: str) -> List[str]:
    """根据中英文标点将文本切分成句子。保留分隔符，避免在句子中间切断。"""
    delimiters = r'([。！？；\.!\?;\n])'
    parts = re.split(delimiters, text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append(parts[i] + parts[i+1])
    if len(parts) % 2 == 1 and parts[-1]:
        sentences.append(parts[-1])
    return [s.strip() for s in sentences if s.strip()]

def get_item_text(item: dict) -> str:
    """根据类型提取项的完整文本内容。"""
    itype = item.get("type")
    if itype == "text":
        return item.get("text", "")
    elif itype == "list":
        return "\n".join(item.get("list_items", []))
    elif itype == "table":
        caption = " ".join(item.get("table_caption", [])) if item.get("table_caption") else ""
        body = item.get("table_body", "")
        return f"{caption}\n{body}" if caption else body
    return ""

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
    """将 OCR 解析结果（JSON 格式）提交到 RAG 数据库"""
    # 1. 获取 OCR 任务信息
    stmt = select(OcrDocument).where(OcrDocument.id == ocr_id)
    result = await db.execute(stmt)
    ocr_doc = result.scalar_one_or_none()
    
    if not ocr_doc:
        raise HTTPException(status_code=404, detail="OCR 任务未找到")
    if ocr_doc.ocr_status != "已识别":
        raise HTTPException(status_code=400, detail="OCR 任务尚未完成识别")
    if not ocr_doc.json_file_url:
        raise HTTPException(status_code=400, detail="OCR 任务没有 JSON 结果文件，请先重新识别")
    if ocr_doc.rag_status == "已提交":
        raise HTTPException(status_code=400, detail="任务已提交过 RAG 数据库")

    try:
        # 2. 下载 JSON 内容
        async with httpx.AsyncClient() as client:
            resp = await client.get(ocr_doc.json_file_url, timeout=60.0)
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail="无法下载 OCR JSON 结果文件")
            data = resp.json()

        # 3. 按语句完整性切分 JSON 内容为 Trunk
        # 获取系统配置的 Trunk 建议字数
        cfg_stmt = select(SystemConfig).where(SystemConfig.config_key == "ocr_trunk_size")
        cfg_res = await db.execute(cfg_stmt)
        cfg_val = cfg_res.scalar_one_or_none()
        suggested_size = int(cfg_val.config_value) if cfg_val else 512
        
        allowed_types = ["text", "list", "table"]
        # 排除页眉、页脚、脚注等
        excluded_types = ["header", "footer", "page_header", "page_footer", "footnote", "page_number"]
        
        trunks = []
        current_trunk_text = ""
        current_trunk_pages = set()

        for item in data:
            itype = item.get("type")
            # 过滤掉不参与 RAG 的类型以及页眉页脚
            if itype in excluded_types or itype not in allowed_types:
                continue

            item_text = get_item_text(item).strip()
            if not item_text:
                continue

            page_idx = item.get("page_idx")
            
            # 拼接内容
            if current_trunk_text:
                current_trunk_text += "\n" + item_text
            else:
                current_trunk_text = item_text
            
            if page_idx is not None:
                current_trunk_pages.add(page_idx)
            
            # 如果当前积攒的内容已达到或超过建议限制，则作为一个 Trunk
            if len(current_trunk_text) >= suggested_size:
                trunks.append({
                    "content": current_trunk_text,
                    "page_indices": sorted(list(current_trunk_pages))
                })
                current_trunk_text = ""
                current_trunk_pages = set()

        # 处理最后剩余的内容
        if current_trunk_text:
            trunks.append({
                "content": current_trunk_text,
                "page_indices": sorted(list(current_trunk_pages))
            })

        # 4. 创建关联文档记录
        doc_filename = ocr_doc.filename or f"OCR_{ocr_id.hex[:6]}.pdf"
        
        # 检查是否已存在同名文档（如果需要唯一性）
        doc = Document(
            filename=f"OCR_{ocr_id.hex[:6]}_{doc_filename}",
            oss_key=f"ocr_raw/{ocr_doc.original_file_url.split('/')[-1].split('?')[0]}",
            uploader=ocr_doc.uploader,
            kb_type=kb_type,
            region_level=region_level,
            province=province,
            city=city
        )
        db.add(doc)
        await db.flush()

        # 5. 插入 Clauses
        inserted_count = 0
        for trunk in trunks:
            content_text = trunk["content"]
            # 提取 chapter_path: 采用文本块的前 20 个字符作为简单标题，或者固定为 "OCR 解析内容"
            simple_title = content_text[:30].replace('\n', ' ') + "..." if len(content_text) > 30 else content_text
            
            embedding = rag_service.get_embedding(content_text)
            clause = Clause(
                doc_id=doc.id,
                kb_type=kb_type,
                chapter_path=simple_title,
                content=content_text,
                page_number=(trunk["page_indices"][0] + 1) if trunk["page_indices"] and trunk["page_indices"][0] is not None else None,
                creator=current_user["username"],
                import_method="ocr 导入",
                embedding=embedding,
                is_verified=True,
                region_level=region_level,
                province=province,
                city=city
            )
            db.add(clause)
            inserted_count += 1

        # 6. 更新 OCR 任务状态
        ocr_doc.rag_status = "已提交"
        ocr_doc.submit_time = get_beijing_time()
        ocr_doc.submitter = current_user["username"]
        
        await db.commit()
        await db.refresh(ocr_doc)
        logger.info(f"[OCR RAG] Task {ocr_id} processed: {inserted_count} clauses inserted.")
        return ocr_doc
        
    except Exception as e:
        await db.rollback()
        logger.error(f"[OCR RAG] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"提交 RAG 失败: {str(e)}")
