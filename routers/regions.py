from typing import List, Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db
from ..models import Region
from ..schemas import RegionOut

router = APIRouter(prefix="/regions", tags=["区域信息"])

@router.get("/provinces", response_model=List[RegionOut])
async def get_provinces(db: AsyncSession = Depends(get_db)):
    """获取所有省份/直辖市"""
    stmt = select(Region).where(Region.level == 1).order_by(Region.id)
    result = await db.execute(stmt)
    return result.scalars().all()

@router.get("/{parent_id}/cities", response_model=List[RegionOut])
async def get_cities(parent_id: int, db: AsyncSession = Depends(get_db)):
    """获取指定省份下的所有城市"""
    stmt = select(Region).where(Region.parent_id == parent_id, Region.level == 2).order_by(Region.id)
    result = await db.execute(stmt)
    return result.scalars().all()
