import uuid
import httpx
import zipfile
import io
import logging
import os
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import OcrDocument, get_beijing_time
from .mineru_service import mineru_service
from .oss_service import oss_service

logger = logging.getLogger("ocr_process_service")

async def process_mineru_result(task: OcrDocument, zip_url: str, db: AsyncSession):
    """
    下载 MinerU 结果 ZIP，提取 JSON 和 MD，分别上传到 OSS，最后上传 ZIP
    """
    try:
        logger.info(f"[OCR Process] Starting download for task {task.id} from {zip_url}...")
        async with httpx.AsyncClient() as client:
            resp = await client.get(zip_url, timeout=240.0)
            if resp.status_code != 200:
                logger.error(f"[OCR Process] Failed to download ZIP for {task.id}: HTTP {resp.status_code}")
                return False

        zip_content = resp.content
        logger.info(f"[OCR Process] Download complete ({len(zip_content)} bytes) for task {task.id}")

        # 使用记录 ID 作为基础文件名
        base_filename = str(task.id)
        
        json_url = None
        md_url = None
        
        # 解析 ZIP 内容
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                file_list = zf.namelist()
                logger.info(f"[OCR Process] ZIP contains {len(file_list)} files: {file_list}")
                
                # 查找 JSON 文件 (优先 content_list.json)
                json_files = [n for n in file_list if n.endswith('.json')]
                target_json = next((n for n in json_files if 'content_list' in n), None)
                if not target_json and json_files:
                    target_json = json_files[0]
                
                if target_json:
                    with zf.open(target_json) as f:
                        json_data = f.read()
                        oss_json_name = f"{base_filename}_content_list.json"
                        logger.info(f"[OCR Process] Uploading JSON to OSS: {oss_json_name}")
                        json_key = oss_service.upload_file(json_data, oss_json_name, directory="ocr_results")
                        json_url = oss_service.get_file_url(json_key)
                
                # 查找 MD 文件
                md_files = [n for n in file_list if n.endswith('.md')]
                target_md = next((n for n in md_files if 'full' in n or 'content' in n), None)
                if not target_md and md_files:
                    target_md = md_files[0]
                
                if target_md:
                    with zf.open(target_md) as f:
                        md_data = f.read()
                        oss_md_name = f"{base_filename}.md"
                        logger.info(f"[OCR Process] Uploading MD to OSS: {oss_md_name}")
                        md_key = oss_service.upload_file(md_data, oss_md_name, directory="ocr_results")
                        md_url = oss_service.get_file_url(md_key)

        except Exception as ze:
            logger.error(f"[OCR Process] Error extracting ZIP for task {task.id}: {ze}")
            # 如果解压失败，我们仍然尝试上传 ZIP，但可能无法更新 JSON/MD URL

        # 最后上传 ZIP 文件
        oss_zip_name = f"{base_filename}.zip"
        logger.info(f"[OCR Process] Uploading ZIP to OSS: {oss_zip_name}")
        oss_zip_key = oss_service.upload_file(zip_content, oss_zip_name, directory="ocr_results")
        zip_result_url = oss_service.get_file_url(oss_zip_key)

        # 更新数据库
        task.ocr_status = "已识别"
        task.result_file_url = zip_result_url
        task.json_file_url = json_url
        task.md_file_url = md_url
        task.ocr_time = get_beijing_time()
        
        await db.commit()
        logger.info(f"[OCR Process] Task {task.id} finished successfully.")
        return True

    except Exception as e:
        logger.error(f"[OCR Process] Unexpected error processing task {task.id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
