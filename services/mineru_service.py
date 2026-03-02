import httpx
import logging
from ..config import settings
import json

# 配置日志
logger = logging.getLogger("mineru_service")
logger.setLevel(logging.INFO)

class MinerUService:
    def __init__(self):
        self.api_key = settings.MINERU_API_KEY
        self.base_url = settings.MINERU_BASE_URL.rstrip('/')
    
    async def submit_task(self, file_url: str):
        """
        提交 MinerU OCR 解析任务
        :param file_url: 公开可访问的文件 URL
        :return: task_id
        """
        if not self.api_key:
            raise ValueError("MINERU_API_KEY is not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "url": file_url,
            "is_ocr": True,
            "enable_formula": True,
            "enable_table": True,
            "language": "ch",
            "include_original": False # 解析文件的结果不需要包含原文件
        }
        
        url = f"{self.base_url}/extract/task"
        logger.info(f"[MinerU Submit] Request URL: {url}")
        logger.info(f"[MinerU Submit] Payload: {json.dumps(payload, ensure_ascii=False)}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload, timeout=30.0)
                logger.info(f"[MinerU Submit] Response Status: {response.status_code}")
                
                try:
                    data = response.json()
                    logger.info(f"[MinerU Submit] Response Body: {json.dumps(data, ensure_ascii=False)}")
                except Exception as je:
                    logger.error(f"[MinerU Submit] Failed to parse JSON response: {je}")
                    logger.error(f"[MinerU Submit] Raw Response Text: {response.text[:500]}")
                    raise Exception(f"MinerU API Response is not JSON: {response.text[:200]}")
                
                if response.status_code != 200 or data.get("code") != 0:
                    error_msg = data.get('msg', 'Unknown Error')
                    logger.error(f"[MinerU Submit] API Error Reported: {error_msg}")
                    raise Exception(f"MinerU API Error: {error_msg}")
                
                return data["data"]["task_id"]
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                logger.error(f"[MinerU Submit] Full Traceback:\n{error_detail}")
                raise

    async def get_task_status(self, task_id: str):
        """
        获取 MinerU OCR 解析任务状态
        :param task_id: 任务 ID
        :return: status (str), full_zip_url (str, optional)
        """
        if not self.api_key:
            raise ValueError("MINERU_API_KEY is not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.info(f"[MinerU Status] Checking task: {task_id}")
        
        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}/extract/task/{task_id}"
                logger.info(f"[MinerU Status] Request URL: {url}")
                response = await client.get(url, headers=headers, timeout=10.0)
                logger.info(f"[MinerU Status] Response Status: {response.status_code}")
                
                try:
                    data = response.json()
                    logger.info(f"[MinerU Status] Response Body: {json.dumps(data, ensure_ascii=False)}")
                except Exception as je:
                    logger.error(f"[MinerU Status] Failed to parse JSON response: {je}")
                    logger.error(f"[MinerU Status] Raw Response Text: {response.text[:500]}")
                    raise Exception(f"MinerU API Response is not JSON: {response.text[:200]}")

                if response.status_code != 200 or data.get("code") != 0:
                    error_msg = data.get('msg', 'Unknown Error')
                    logger.error(f"[MinerU Status] API Error Reported: {error_msg}")
                    raise Exception(f"MinerU API Error: {error_msg}")
                
                task_data = data["data"]
                # MinerU API 返回的状态键名为 'state'
                # 可能的状态值有: 'pending', 'parsing', 'done', 'error'
                raw_state = task_data.get("state")
                full_zip_url = task_data.get("full_zip_url")
                
                # 统一映射状态为 success, failed, parsing
                mapped_status = "parsing"
                if raw_state == "done":
                    mapped_status = "success"
                elif raw_state == "error" or raw_state == "failed":
                    mapped_status = "failed"
                elif raw_state == "pending" or raw_state == "parsing":
                    mapped_status = "parsing"
                
                logger.info(f"[MinerU Status] Task {task_id} mapped status: {mapped_status}")
                return mapped_status, full_zip_url
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                logger.error(f"[MinerU Status] Full Traceback for task {task_id}:\n{error_detail}")
                raise

mineru_service = MinerUService()
