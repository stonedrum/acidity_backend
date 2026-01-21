from openai import AsyncOpenAI
from ..config import settings
import json
from typing import List, Dict, Any, AsyncGenerator

class LLMService:
    def __init__(self):
        self.default_client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        # 只有在配置了 DEEPSEEK_API_KEY 时才初始化 deepseek 客户端
        self.deepseek_client = None
        if settings.DEEPSEEK_API_KEY:
            self.deepseek_client = AsyncOpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL
            )

    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        stream: bool = False
    ) -> Any:
        """
        Generic chat completion that supports both streaming and non-streaming.
        """
        target_model = model or settings.LLM_MODEL
        
        # 选择客户端：如果设置了 DEEPSEEK_API_KEY 且模型包含 deepseek，则使用专门的 deepseek 客户端
        # 否则一律使用默认客户端（如阿里云 DashScope）
        client = self.default_client
        if "deepseek" in target_model.lower() and self.deepseek_client:
            client = self.deepseek_client
        
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
