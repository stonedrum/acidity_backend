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
        # 向量模型加载
        # 如果设置了 VECTOR_MODEL_PATH，优先使用自定义路径；否则使用 VECTOR_MODEL_NAME
        vector_model_path = settings.VECTOR_MODEL_PATH if settings.VECTOR_MODEL_PATH else settings.VECTOR_MODEL_NAME
        print(f"[RAGService] 加载向量模型: {vector_model_path}")
        # 使用 local_files_only=True 强制使用本地文件（不下载）
        self.embedding_model = SentenceTransformer(vector_model_path, local_files_only=True)
        
        # 重排模型加载
        # 如果设置了 RERANK_MODEL_PATH，优先使用自定义路径；否则使用 RERANK_MODEL_NAME
        rerank_model_path = settings.RERANK_MODEL_PATH if settings.RERANK_MODEL_PATH else settings.RERANK_MODEL_NAME
        print(f"[RAGService] 加载重排模型: {rerank_model_path}")
        # 注意：CrossEncoder 可能不支持 local_files_only，但环境变量应该能阻止网络请求
        try:
            self.rerank_model = CrossEncoder(rerank_model_path, local_files_only=True)
        except Exception as e:
            # 如果 local_files_only 不支持，尝试不使用该参数，但环境变量应该已经禁用了网络
            print(f"[RAGService] 警告: 使用 local_files_only 加载重排模型失败，尝试不使用该参数: {e}")
            self.rerank_model = CrossEncoder(rerank_model_path)

    def get_embedding(self, text):
        return self.embedding_model.encode(text).tolist()

    async def search_and_rerank(self, query, db_session, return_initial_results=False):
        """
        搜索并重排
        :param query: 查询文本
        :param db_session: 数据库会话
        :param return_initial_results: 是否返回初始向量匹配结果
        :return: 如果 return_initial_results=True，返回 (initial_results, reranked_results)
                 否则只返回 reranked_results (Top 3)
        """
        from ..models import Clause
        from sqlalchemy import select
        from pgvector.sqlalchemy import Vector
        
        query_embedding = self.get_embedding(query)
        
        # 1. Vector Match (Initial Screening)
        # We use cosine similarity (1 - <=> in pgvector)
        # user wants similarity > 0.3, Top 10
        # pgvector <=> is cosine distance, so similarity = 1 - distance
        # similarity > 0.3 means distance < 0.7
        
        # Filter by similarity > 0.3 in SQL (equivalent to distance < 0.7)
        # In PostgreSQL: WHERE (1 - (embedding <=> query_embedding)) > 0.3
        # Which is equivalent to: WHERE (embedding <=> query_embedding) < 0.7
        distance_threshold = 0.7  # 1 - 0.3 = 0.7
        
        # Filter candidates by cosine distance threshold in SQL
        # pgvector's cosine_distance() can be used in WHERE clause
        stmt = (
            select(Clause)
            .where(
                Clause.embedding.cosine_distance(query_embedding) < distance_threshold
            )
            .order_by(Clause.embedding.cosine_distance(query_embedding))
            .limit(10)
        )
        result = await db_session.execute(stmt)
        candidates = result.scalars().all()
        
        if not candidates:
            if return_initial_results:
                return [], []
            return []

        # 查询文档信息（用于获取文档名称）
        from ..models import Document
        doc_ids = set(c.doc_id for c in candidates)
        doc_map = {}
        if doc_ids:
            doc_stmt = select(Document).where(Document.id.in_(doc_ids))
            doc_result = await db_session.execute(doc_stmt)
            docs = doc_result.scalars().all()
            doc_map = {str(doc.id): doc.filename for doc in docs}

        # 计算初始结果的相似度分数（1 - distance）
        initial_results = []
        for c in candidates:
            # 计算余弦距离
            import numpy as np
            clause_embedding = np.array(c.embedding)
            query_emb = np.array(query_embedding)
            # 计算余弦距离
            dot_product = np.dot(clause_embedding, query_emb)
            norm_clause = np.linalg.norm(clause_embedding)
            norm_query = np.linalg.norm(query_emb)
            cosine_similarity = dot_product / (norm_clause * norm_query) if (norm_clause * norm_query) > 0 else 0.0
            initial_results.append({
                "clause_id": str(c.id),
                "doc_id": str(c.doc_id),
                "doc_name": doc_map.get(str(c.doc_id), "未知文档"),  # 添加文档名称
                "chapter_path": c.chapter_path,
                "content": c.content[:500],  # 只保存前500字符，避免过长
                "score": float(cosine_similarity)
            })

        # 2. Rerank (Refining Top 3)
        pairs = [[query, c.content] for c in candidates]
        scores = self.rerank_model.predict(pairs)
        
        # Sort candidates by rerank score
        ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        # Top 3 and similarity score > 0.4
        top3_results = [x for x in ranked_results if x[1] > 0.7]
        if len(top3_results) > 3:
            top3_results = ranked_results[:3]

        if return_initial_results:
            # 返回初始结果和重排结果
            reranked_data = []
            for c, score in top3_results:
                reranked_data.append({
                    "clause_id": str(c.id),
                    "doc_id": str(c.doc_id),
                    "doc_name": doc_map.get(str(c.doc_id), "未知文档"),  # 添加文档名称
                    "chapter_path": c.chapter_path,
                    "content": c.content[:500],  # 只保存前500字符
                    "rerank_score": float(score)
                })
            return initial_results, reranked_data
        
        # 返回 Top 3（兼容原有接口）
        return top3_results

rag_service = RAGService()
