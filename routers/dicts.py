from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db
from ..models import DictType, DictData
from ..schemas import (
    DictDataOut, DictTypeOut, DictTypeCreate, DictTypeUpdate,
    DictDataCreate, DictDataUpdate
)
from ..auth import get_current_user

router = APIRouter(tags=["数据字典"])

@router.get("/dicts/{type_name}", response_model=List[DictDataOut])
async def get_dict_by_type(type_name: str, db: AsyncSession = Depends(get_db)):
    """根据类型名称获取字典项（如 /dicts/kb_type）"""
    stmt = (
        select(DictData)
        .join(DictType)
        .where(DictType.type_name == type_name, DictData.is_active == True)
        .order_by(DictData.sort_order)
    )
    result = await db.execute(stmt)
    return result.scalars().all()

@router.get("/dict-types", response_model=List[DictTypeOut])
async def list_dict_types(db: AsyncSession = Depends(get_db), current_user: str = Depends(get_current_user)):
    """列出所有字典类型（管理端使用）"""
    stmt = select(DictType)
    result = await db.execute(stmt)
    types = result.scalars().all()
    
    out = []
    for t in types:
        data_stmt = select(DictData).where(DictData.type_id == t.id).order_by(DictData.sort_order)
        data_result = await db.execute(data_stmt)
        t_data = data_result.scalars().all()
        out.append(DictTypeOut(
            id=t.id,
            type_name=t.type_name,
            description=t.description,
            data=[DictDataOut.from_orm(d) for d in t_data]
        ))
    return out

@router.post("/dict-types", response_model=DictTypeOut)
async def create_dict_type(
    type_data: DictTypeCreate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    stmt = select(DictType).where(DictType.type_name == type_data.type_name)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Dict type already exists")
    
    new_type = DictType(**type_data.dict())
    db.add(new_type)
    await db.commit()
    await db.refresh(new_type)
    return DictTypeOut(id=new_type.id, type_name=new_type.type_name, description=new_type.description, data=[])

@router.put("/dict-types/{type_id}", response_model=DictTypeOut)
async def update_dict_type(
    type_id: UUID,
    type_data: DictTypeUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    stmt = select(DictType).where(DictType.id == type_id)
    result = await db.execute(stmt)
    dict_type = result.scalar_one_or_none()
    if not dict_type:
        raise HTTPException(status_code=404, detail="Dict type not found")
    
    for key, value in type_data.dict(exclude_unset=True).items():
        setattr(dict_type, key, value)
    
    await db.commit()
    await db.refresh(dict_type)
    
    data_stmt = select(DictData).where(DictData.type_id == dict_type.id).order_by(DictData.sort_order)
    data_result = await db.execute(data_stmt)
    t_data = data_result.scalars().all()
    
    return DictTypeOut(
        id=dict_type.id,
        type_name=dict_type.type_name,
        description=dict_type.description,
        data=[DictDataOut.from_orm(d) for d in t_data]
    )

@router.delete("/dict-types/{type_id}")
async def delete_dict_type(
    type_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    stmt = select(DictType).where(DictType.id == type_id)
    result = await db.execute(stmt)
    dict_type = result.scalar_one_or_none()
    if not dict_type:
        raise HTTPException(status_code=404, detail="Dict type not found")
    
    if dict_type.type_name == "kb_type":
        raise HTTPException(status_code=400, detail="Core dict type 'kb_type' cannot be deleted")
        
    await db.delete(dict_type)
    await db.commit()
    return {"message": "Dict type deleted"}

@router.post("/dict-data/{type_id}", response_model=DictDataOut)
async def create_dict_data(
    type_id: UUID,
    data_item: DictDataCreate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    stmt = select(DictType).where(DictType.id == type_id)
    result = await db.execute(stmt)
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Dict type not found")
        
    new_data = DictData(type_id=type_id, **data_item.dict())
    db.add(new_data)
    await db.commit()
    await db.refresh(new_data)
    return new_data

@router.put("/dict-data/{data_id}", response_model=DictDataOut)
async def update_dict_data(
    data_id: UUID,
    data_item: DictDataUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    stmt = select(DictData).where(DictData.id == data_id)
    result = await db.execute(stmt)
    db_data = result.scalar_one_or_none()
    if not db_data:
        raise HTTPException(status_code=404, detail="Dict data not found")
    
    for key, value in data_item.dict(exclude_unset=True).items():
        setattr(db_data, key, value)
    
    await db.commit()
    await db.refresh(db_data)
    return db_data

@router.delete("/dict-data/{data_id}")
async def delete_dict_data(
    data_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    stmt = select(DictData).where(DictData.id == data_id)
    result = await db.execute(stmt)
    db_data = result.scalar_one_or_none()
    if not db_data:
        raise HTTPException(status_code=404, detail="Dict data not found")
        
    await db.delete(db_data)
    await db.commit()
    return {"message": "Dict data deleted"}
