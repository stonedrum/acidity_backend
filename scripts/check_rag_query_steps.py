#!/usr/bin/env python3
"""检查 chat_query_logs 表是否有 rag_query_steps 列，并打印最近一条日志的该字段（用于验证查询日志详情页 RAG 拼接过程）。
运行方式（在项目根目录）: python -m backend.scripts.check_rag_query_steps
或: cd backend && PYTHONPATH=. python scripts/check_rag_query_steps.py
"""
import asyncio
import sys
import os
# 保证 backend 作为包可被导入（项目根目录或 backend 目录运行）
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
# 以 backend 包方式导入
from sqlalchemy import text, select
from backend.database import engine, AsyncSessionLocal
from backend.models import ChatQueryLog


async def main():
    async with engine.begin() as conn:
        r = await conn.execute(text("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'chat_query_logs' AND column_name = 'rag_query_steps'
        """))
        row = r.fetchone()
        if not row:
            print("表 chat_query_logs 中不存在列 rag_query_steps，正在添加...")
            await conn.execute(text("ALTER TABLE chat_query_logs ADD COLUMN IF NOT EXISTS rag_query_steps JSONB"))
            print("✓ 已添加列 rag_query_steps。请重新发起一次「智能问答」后再在查询日志中查看该次「查看详情」。")
            return
    print("✓ 表 chat_query_logs 存在列 rag_query_steps")

    async with AsyncSessionLocal() as session:
        r = await session.execute(
            select(ChatQueryLog).order_by(ChatQueryLog.query_time.desc()).limit(3)
        )
        logs = r.scalars().all()
    if not logs:
        print("当前没有查询日志记录。请先在智能问答中发起一次对话后再查看日志详情。")
        return
    for i, log in enumerate(logs):
        has_steps = log.rag_query_steps is not None and len(log.rag_query_steps or []) > 0
        print(f"  最近第 {i+1} 条: id={log.id}, query_time={log.query_time}, rag_query_steps={'有('+str(len(log.rag_query_steps))+' 步)' if has_steps else '无(null 或空)'}")
    if not any((log.rag_query_steps and len(log.rag_query_steps) > 0) for log in logs):
        print("\n说明: 以上记录均为旧数据（该功能上线前产生），rag_query_steps 为空。请先进行一次新的智能问答，再在「查询日志」中点击该次对话的「查看详情」，即可看到 RAG 拼接查询过程。")


if __name__ == "__main__":
    asyncio.run(main())
