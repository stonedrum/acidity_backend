from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import List, Optional
from uuid import UUID

from ..database import get_db
from ..models import User
from ..schemas import UserOut, UserUpdate, PaginatedUsers, UserCreate, PasswordChange
from ..auth import get_password_hash, check_role, get_current_user, verify_password

router = APIRouter(tags=["用户管理"])

@router.put("/users/me/password")
async def change_my_password(
    data: PasswordChange,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    stmt = select(User).where(User.username == current_user["username"])
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not verify_password(data.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="旧密码错误")
    
    user.password_hash = get_password_hash(data.new_password)
    await db.commit()
    return {"status": "ok", "message": "密码修改成功"}

@router.get("/me", response_model=UserOut)
async def get_my_info(current_user: dict = Depends(get_current_user)):
    """获取当前用户信息（用于验证 token 有效性）"""
    return current_user

@router.get("/users", response_model=PaginatedUsers)
async def list_users(
    page: int = 1,
    page_size: int = 15,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin"]))
):
    query = select(User).order_by(desc(User.id))
    
    # 分页逻辑
    count_stmt = select(func.count()).select_from(User)
    count_res = await db.execute(count_stmt)
    total = count_res.scalar()
    
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    users = result.scalars().all()
    
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    
    return PaginatedUsers(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        items=users
    )

@router.post("/users", response_model=UserOut)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin"]))
):
    stmt = select(User).where(User.username == user_data.username)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        password_hash=hashed_password,
        role=user_data.role or "user"
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

@router.put("/users/{user_id}", response_model=UserOut)
async def update_user(
    user_id: UUID,
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin"]))
):
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user_data.username is not None:
        user.username = user_data.username
    if user_data.role is not None:
        user.role = user_data.role
    if user_data.password is not None:
        user.password_hash = get_password_hash(user_data.password)
        
    await db.commit()
    await db.refresh(user)
    return user

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin"]))
):
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 不允许删除自己
    # 注意：这里 current_user["username"] 需要从 token 获取
    if user.username == current_user["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    await db.delete(user)
    await db.commit()
    return {"status": "ok", "message": "User deleted"}

@router.post("/users/{user_id}/reset-password")
async def reset_password(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(check_role(["sysadmin"]))
):
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.password_hash = get_password_hash("123456")
    await db.commit()
    return {"status": "ok", "message": "Password reset to 123456"}
