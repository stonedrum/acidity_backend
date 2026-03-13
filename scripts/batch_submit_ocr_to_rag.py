import asyncio
import os
import sys
import httpx
import uuid
import logging
from sqlalchemy import select
from typing import List

# 保证 backend 作为包可被导入
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.database import AsyncSessionLocal, engine
from backend.models import OcrDocument, Document, Clause, SystemConfig, get_beijing_time
from backend.services.rag_service import rag_service

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("batch_submit_ocr")

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

async def process_one_document(ocr_doc: OcrDocument, db):
    """处理单个 OCR 任务提交到 RAG"""
    ocr_id = ocr_doc.id
    final_kb_type = ocr_doc.kb_type
    region_level = "全国"
    province = ""
    city = ""
    
    logger.info(f"开始处理任务: {ocr_id}, 文件名: {ocr_doc.filename}, 类型: {final_kb_type}")

    try:
        # 1. 下载 JSON 内容
        async with httpx.AsyncClient() as client:
            resp = await client.get(ocr_doc.json_file_url, timeout=60.0)
            if resp.status_code != 200:
                logger.error(f"任务 {ocr_id} 下载 JSON 失败: {resp.status_code}")
                return False
            data = resp.json()

        # 2. 获取分块大小配置
        cfg_stmt = select(SystemConfig).where(SystemConfig.config_key == "ocr_trunk_size")
        cfg_res = await db.execute(cfg_stmt)
        cfg_val = cfg_res.scalar_one_or_none()
        suggested_size = int(cfg_val.config_value) if cfg_val else 512
        
        # 兼容性处理
        if isinstance(data, dict):
            if "content_list" in data:
                data = data["content_list"]
            elif "data" in data:
                data = data["data"]
        
        if not isinstance(data, list):
            logger.error(f"任务 {ocr_id} JSON 格式异常")
            return False

        allowed_types = ["text", "list", "table"]
        excluded_types = ["header", "footer", "page_header", "page_footer", "footnote", "page_number"]
        
        trunks = []
        current_trunk_text = ""
        current_trunk_pages = set()

        def flush_trunk():
            nonlocal current_trunk_text, current_trunk_pages
            if current_trunk_text:
                trunks.append({
                    "content": current_trunk_text.strip(),
                    "page_indices": sorted(list(current_trunk_pages))
                })
                current_trunk_text = ""
                current_trunk_pages = set()

        for item in data:
            itype = item.get("type")
            if itype in excluded_types or itype not in allowed_types:
                continue

            item_text = get_item_text(item).strip()
            if not item_text:
                continue

            page_idx = item.get("page_idx")
            if current_trunk_text:
                current_trunk_text += "\n" + item_text
            else:
                current_trunk_text = item_text
            
            if page_idx is not None:
                try:
                    current_trunk_pages.add(int(page_idx))
                except (ValueError, TypeError):
                    pass
            
            if len(current_trunk_text) >= suggested_size:
                flush_trunk()

        flush_trunk()

        # 3. 创建关联文档记录
        doc_filename = ocr_doc.filename or f"OCR_{ocr_id.hex[:6]}.pdf"
        full_doc_filename = f"OCR_{ocr_id.hex[:6]}_{doc_filename}"
        
        # 检查是否已存在（由于文件名带了随机 ID，理论上不会冲突）
        doc = Document(
            filename=full_doc_filename,
            oss_key=f"ocr_raw/{ocr_doc.original_file_url.split('/')[-1].split('?')[0]}",
            uploader=ocr_doc.uploader,
            kb_type=final_kb_type,
            region_level=region_level,
            province=province,
            city=city
        )
        db.add(doc)
        await db.flush()

        # 4. 插入 Clauses
        inserted_count = 0
        for trunk in trunks:
            content_text = trunk["content"]
            simple_title = content_text[:30].replace('\n', ' ') + "..." if len(content_text) > 30 else content_text
            
            embedding = rag_service.get_embedding(content_text)
            clause = Clause(
                doc_id=doc.id,
                kb_type=final_kb_type,
                chapter_path=simple_title,
                content=content_text,
                page_number=(trunk["page_indices"][0] + 1) if trunk["page_indices"] and trunk["page_indices"][0] is not None else None,
                creator=ocr_doc.uploader or "system_batch",
                import_method="ocr 导入",
                embedding=embedding,
                is_verified=True,
                region_level=region_level,
                province=province,
                city=city
            )
            db.add(clause)
            inserted_count += 1

        # 5. 更新 OCR 任务状态
        ocr_doc.rag_status = "已提交"
        ocr_doc.submit_time = get_beijing_time()
        ocr_doc.submitter = "system_batch"
        
        await db.commit()
        logger.info(f"任务 {ocr_id} 处理成功: 插入了 {inserted_count} 条知识片段")
        return True
        
    except Exception as e:
        await db.rollback()
        logger.error(f"任务 {ocr_id} 处理失败: {e}")
        return False

async def main():
    async with AsyncSessionLocal() as session:
        # 查询符合条件的记录
        stmt = select(OcrDocument).where(
            OcrDocument.ocr_status == "已识别",
            OcrDocument.rag_status == "未提交",
            OcrDocument.kb_type != None,
            OcrDocument.kb_type != ""
        )
        result = await session.execute(stmt)
        tasks = result.scalars().all()
        
        if not tasks:
            logger.info("没有找到符合条件的待提交 OCR 任务。")
            return

        logger.info(f"找到 {len(tasks)} 条符合条件的任务，开始批量处理...")
        
        success_count = 0
        fail_count = 0
        
        for task in tasks:
            if await process_one_document(task, session):
                success_count += 1
            else:
                fail_count += 1
        
        logger.info(f"批量处理完成！成功: {success_count}, 失败: {fail_count}")

if __name__ == "__main__":
    asyncio.run(main())
