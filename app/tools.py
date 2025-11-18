# # app/tools.py

# from typing import List
# from langchain_core.documents import Document   # ✅ FIXED IMPORT
# from tavily import TavilyClient

# class RetrieverTool:
#     def __init__(self, retriever):
#         self.retriever = retriever

#     def run(self, query: str, k: int = 3) -> List[Document]:
#         # returns list of langchain Document
#         if hasattr(self.retriever, "get_relevant_documents"):
#             return self.retriever.get_relevant_documents(query)
#         return self.retriever(query)


# def calc_tool(expression: str) -> str:
#     try:
#         allowed = {"__builtins__": {}}
#         result = eval(expression, allowed, {})
#         return str(result)
#     except Exception as e:
#         return f"calc_error: {e}"


# def summarizer_tool(docs) -> str:
#     texts = []
#     for d in docs:
#         if hasattr(d, "page_content"):
#             texts.append(d.page_content)
#         else:
#             texts.append(str(d))
#     combined = "\n\n".join(texts)
#     return combined[:4000]


# class WebSearchTool:
#     def __init__(self, api_key: str):
#         self.client = TavilyClient(api_key)

#     def run(self, query: str, k: int = 5):
#         """Perform real-time web search."""
#         try:
#             result = self.client.search(query=query, max_results=k)
#             # return combined text
#             texts = [item["content"] for item in result.get("results", [])]
#             return "\n\n".join(texts)
#         except Exception as e:
#             return f"web_search_error: {e}"









# app/tools.py
import os
from typing import List
from langchain_core.documents import Document
from tavily import TavilyClient

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore


# ------------------------------
# PINECONE INITIALIZER
# ------------------------------
def init_pinecone_index(api_key: str, index_name: str, embedding_model: str):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    pc = Pinecone(api_key=api_key)
    vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name
    )
    return vectorstore


# ------------------------------
# RETRIEVER TOOL
# ------------------------------
class RetrieverTool:
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query: str, k: int = 5) -> List[Document]:
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(query)
        return self.retriever(query)


# ------------------------------
# WEB SEARCH TOOL
# ------------------------------
class WebSearchTool:
    def __init__(self, api_key: str | None):
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set")
        self.client = TavilyClient(api_key)

    def run(self, query: str, k: int = 5) -> str:
        resp = self.client.search(query=query, max_results=k)
        texts = [
            r.get("content") or r.get("snippet") or ""
            for r in resp.get("results", [])
        ]
        return "\n\n".join(texts)
