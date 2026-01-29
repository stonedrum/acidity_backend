from typing import List
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db
from ..models import Prompt, get_beijing_time
from ..schemas import PromptOut, PromptCreate, PromptUpdate
from ..auth import get_current_user, check_role

router = APIRouter(tags=["提示词管理"])

@router.get("/prompts", response_model=List[PromptOut])
async def list_prompts(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin"]))
):
    stmt = select(Prompt).order_by(Prompt.name)
    result = await db.execute(stmt)
    return result.scalars().all()

@router.post("/prompts", response_model=PromptOut)
async def create_prompt(
    prompt_data: PromptCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin"]))
):
    stmt = select(Prompt).where(Prompt.name == prompt_data.name)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"Prompt with name '{prompt_data.name}' already exists")
    
    new_prompt = Prompt(
        name=prompt_data.name,
        template=prompt_data.template,
        description=prompt_data.description,
        is_active=prompt_data.is_active
    )
    db.add(new_prompt)
    await db.commit()
    await db.refresh(new_prompt)
    return new_prompt

@router.put("/prompts/{prompt_id}", response_model=PromptOut)
async def update_prompt(
    prompt_id: UUID,
    prompt_data: PromptUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin"]))
):
    stmt = select(Prompt).where(Prompt.id == prompt_id)
    result = await db.execute(stmt)
    prompt_obj = result.scalar_one_or_none()
    if not prompt_obj:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    if prompt_data.name is not None:
        if prompt_data.name != prompt_obj.name:
            name_check = await db.execute(select(Prompt).where(Prompt.name == prompt_data.name))
            if name_check.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="New name already exists")
        prompt_obj.name = prompt_data.name
    
    if prompt_data.template is not None:
        prompt_obj.template = prompt_data.template
        prompt_obj.version += 1
    
    if prompt_data.description is not None:
        prompt_obj.description = prompt_data.description
        
    if prompt_data.is_active is not None:
        prompt_obj.is_active = prompt_data.is_active
    
    prompt_obj.updated_at = get_beijing_time()
    await db.commit()
    await db.refresh(prompt_obj)
    return prompt_obj

@router.delete("/prompts/{prompt_id}")
async def delete_prompt(
    prompt_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin"]))
):
    stmt = select(Prompt).where(Prompt.id == prompt_id)
    result = await db.execute(stmt)
    prompt_obj = result.scalar_one_or_none()
    if not prompt_obj:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    if prompt_obj.name == "rag_system_prompt":
        raise HTTPException(status_code=400, detail="Cannot delete core system prompt")
    
    await db.delete(prompt_obj)
    await db.commit()
    return {"message": "Prompt deleted successfully"}
