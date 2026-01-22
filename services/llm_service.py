from openai import AsyncOpenAI
from ..config import settings
from sqlalchemy import select
from ..database import AsyncSessionLocal
from ..models import SystemConfig
import json
from typing import List, Dict, Any, AsyncGenerator

class LLMService:
    def __init__(self):
        self.default_client = None
        self.deepseek_client = None
        self.config_cache = {}
        # 初始化时标记需要刷新
        self.initialized = False

    async def _ensure_initialized(self):
        """确保配置已从数据库加载"""
        if not self.initialized:
            async with AsyncSessionLocal() as db:
                await self.refresh_configs(db)
            self.initialized = True

    async def refresh_configs(self, db):
        """从数据库刷新配置并重新初始化客户端"""
        stmt = select(SystemConfig)
        result = await db.execute(stmt)
        configs = result.scalars().all()
        
        config_map = {c.config_key: c.config_value for c in configs}
        self.config_cache = config_map

        # 获取配置，如果数据库没有，则回退到 settings
        llm_api_key = config_map.get("llm_api_key", settings.LLM_API_KEY)
        llm_base_url = config_map.get("llm_base_url", settings.LLM_BASE_URL)
        
        deepseek_api_key = config_map.get("deepseek_api_key", settings.DEEPSEEK_API_KEY)
        deepseek_base_url = config_map.get("deepseek_base_url", settings.DEEPSEEK_BASE_URL)

        # 初始化/更新默认客户端 (Qwen)
        self.default_client = AsyncOpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url
        )

        # 初始化/更新 DeepSeek 客户端
        if deepseek_api_key:
            self.deepseek_client = AsyncOpenAI(
                api_key=deepseek_api_key,
                base_url=deepseek_base_url
            )
        else:
            self.deepseek_client = None
            
        print("[LLMService] 配置已刷新")

    def get_actual_model_info(self, requested_model: str = None):
        """
        根据传参或数据库配置，解析出最终使用的 [模型名称, 客户端类型]
        """
        db_system_default = self.config_cache.get("system_default_model", "qwen")
        
        # 1. 确定搜索基础 (优先使用传参，否则使用系统默认设置)
        base_choice = requested_model or db_system_default
        
        # 2. 映射逻辑
        if base_choice.lower() in ["qwen", "llm_model", "模型 a"]:
            actual_name = self.config_cache.get("llm_model", settings.LLM_MODEL)
            client_type = "qwen"
        elif base_choice.lower() in ["deepseek", "deepseek_model", "模型 b"]:
            actual_name = self.config_cache.get("deepseek_model", "deepseek-v3")
            client_type = "deepseek"
        else:
            # 如果是具体模型名，通过关键字判断客户端
            actual_name = base_choice
            client_type = "deepseek" if "deepseek" in base_choice.lower() else "qwen"
            
        return actual_name, client_type

    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        stream: bool = False
    ) -> Any:
        """
        Generic chat completion that supports both streaming and non-streaming.
        """
        await self._ensure_initialized()
        
        # 获取最终执行的模型名和对应的客户端类型
        target_model, client_type = self.get_actual_model_info(model)
        
        # 选择客户端
        if client_type == "deepseek":
            if not self.deepseek_client:
                raise Exception(f"DeepSeek 客户端未初始化，请检查后台配置。")
            client = self.deepseek_client
        else:
            if not self.default_client:
                raise Exception(f"默认 LLM 客户端未初始化，请检查后台配置。")
            client = self.default_client
        
        response = await client.chat.completions.create(
            model=target_model,
            messages=messages,
            stream=stream,
            temperature=0.7,
            max_tokens=2000
        )

        response = await client.chat.completions.create(
            model=target_model,
            messages=messages,
            stream=stream,
            temperature=0.7,
            max_tokens=2000
        )
        
        if stream:
            return self._stream_generator(response)
        else:
            return response.choices[0].message.content

    async def _stream_generator(self, response) -> AsyncGenerator[str, None]:
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

llm_service = LLMService()
