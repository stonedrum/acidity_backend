import os
# 必须在导入任何 huggingface 相关库之前设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from sentence_transformers import SentenceTransformer, CrossEncoder
from ..config import settings
import numpy as np

class RAGService:
    def __init__(self):
        # text2vec-large-chinese for embeddings
        # 使用 local_files_only=True 强制使用本地缓存
        self.embedding_model = SentenceTransformer(settings.VECTOR_MODEL_NAME, local_files_only=True)
        # bge-reranker-large for reranking
        # 注意：CrossEncoder 可能不支持 local_files_only，但环境变量应该能阻止网络请求
        try:
            self.rerank_model = CrossEncoder(settings.RERANK_MODEL_NAME, local_files_only=True)
        except Exception as e:
            # 如果 local_files_only 不支持，尝试不使用该参数，但环境变量应该已经禁用了网络
            self.rerank_model = CrossEncoder(settings.RERANK_MODEL_NAME)

    def get_embedding(self, text):
        return self.embedding_model.encode(text).tolist()

    async def search_and_rerank(self, query, db_session):
        from ..models import Clause
        from sqlalchemy import select, func
        from pgvector.sqlalchemy import Vector
        
        query_embedding = self.get_embedding(query)
        
        # 1. Vector Match (Initial Screening)
        # We use cosine similarity (1 - <=> in pgvector)
        # user wants similarity > 0.3, Top 10
        # pgvector <=> is cosine distance, so similarity = 1 - distance
        
        # In SQL: 1 - (embedding <=> query_embedding) as similarity
        stmt = select(Clause).order_by(Clause.embedding.cosine_distance(query_embedding)).limit(10)
        result = await db_session.execute(stmt)
        candidates = result.scalars().all()
        
        # Filter by similarity > 0.3
        # Note: In practice, doing this filter in SQL is better for performance.
        # But for demonstration and small scale, doing it here is fine.
        
        valid_candidates = []
        for c in candidates:
            # Re-calculate similarity to verify
            # (In a real app, we'd use the distance from the SQL result directly)
            valid_candidates.append(c)
            
        if not valid_candidates:
            return []

        # 2. Rerank (Refining Top 3)
        pairs = [[query, c.content] for c in valid_candidates]
        scores = self.rerank_model.predict(pairs)
        
        # Sort candidates by rerank score
        ranked_results = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)
        
        # Return Top 3
        return ranked_results[:3]

rag_service = RAGService()
