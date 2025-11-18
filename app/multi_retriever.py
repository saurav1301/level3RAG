# app/multi_retriever.py

from typing import List
from langchain_core.documents import Document
from app.query_expander import expand_query


class MultiQueryRetriever:
    def __init__(self, base_retriever):
        self.base = base_retriever

    def retrieve(self, query: str, top_k: int = 8) -> List[Document]:
        queries = expand_query(query)
        all_docs = []
        seen_ids = set()

        for q in queries:
            # LCEL retriever uses .invoke(), not .get_relevant_documents()
            docs = self.base.invoke(q)

            for d in docs:
                key = (
                    getattr(d, "metadata", {}).get("source"),
                    d.page_content[:200]
                )
                if key in seen_ids:
                    continue

                seen_ids.add(key)
                all_docs.append(d)

            if len(all_docs) >= top_k:
                break

        return all_docs[:top_k]
