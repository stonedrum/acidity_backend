from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from ..database import get_db
from ..models import SystemConfig
from ..schemas import SystemConfigOut, SystemConfigUpdate
from ..auth import check_role
from ..services.llm_service import llm_service

router = APIRouter(tags=["系统配置"])

@router.get("/configs", response_model=List[SystemConfigOut])
async def list_configs(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin"]))
):
    stmt = select(SystemConfig).order_by(SystemConfig.config_key)
    result = await db.execute(stmt)
    return result.scalars().all()

@router.put("/configs/{config_key}", response_model=SystemConfigOut)
async def update_config(
    config_key: str,
    data: SystemConfigUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin"]))
):
    stmt = select(SystemConfig).where(SystemConfig.config_key == config_key)
    result = await db.execute(stmt)
    config = result.scalar_one_or_none()
    
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    
    config.config_value = data.config_value
    await db.commit()
    await db.refresh(config)
    
    # 动态刷新 LLM 服务的配置
    llm_related_keys = [
        "llm_api_key", "llm_base_url", "llm_model",
        "deepseek_api_key", "deepseek_base_url", "deepseek_model",
        "system_default_model"
    ]
    if config_key in llm_related_keys or config_key.startswith("llm_") or config_key.startswith("deepseek_"):
        await llm_service.refresh_configs(db)
        
    return config
