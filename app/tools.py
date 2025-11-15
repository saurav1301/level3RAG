# app/tools.py

from typing import List
from langchain_core.documents import Document   # ✅ FIXED IMPORT


class RetrieverTool:
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query: str, k: int = 3) -> List[Document]:
        # returns list of langchain Document
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(query)
        return self.retriever(query)


def calc_tool(expression: str) -> str:
    try:
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed, {})
        return str(result)
    except Exception as e:
        return f"calc_error: {e}"


def summarizer_tool(docs) -> str:
    texts = []
    for d in docs:
        if hasattr(d, "page_content"):
            texts.append(d.page_content)
        else:
            texts.append(str(d))
    combined = "\n\n".join(texts)
    return combined[:4000]
