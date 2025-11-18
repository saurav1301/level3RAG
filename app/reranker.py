# app/reranker.py
from typing import List
from langchain_core.documents import Document


# Lightweight cosine-re-ranker using embeddings (cheap & local)
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer('all-MiniLM-L6-v2')




def rerank(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    if not docs:
        return []
    corpus = [d.page_content for d in docs]
    q_emb = model.encode([query])
    doc_embs = model.encode(corpus)
    sims = cosine_similarity(q_emb, doc_embs)[0]
    idxs = list(reversed(sims.argsort()))[:top_k]
    return [docs[i] for i in idxs]