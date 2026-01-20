from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..models import Prompt

async def get_prompt_template(db: AsyncSession, name: str, default_template: str = None) -> str:
    """从数据库获取提示词"""
    stmt = select(Prompt).where(Prompt.name == name, Prompt.is_active == True)
    result = await db.execute(stmt)
    prompt_obj = result.scalar_one_or_none()
    return prompt_obj.template if prompt_obj else default_template
