# app/web_pipeline.py
from typing import List
from app.tools import WebSearchTool


class WebPipeline:
    def __init__(self, tavily_api_key: str | None):
        self.client = WebSearchTool(tavily_api_key) if tavily_api_key else None


    def search(self, query: str, k: int = 5) -> str:
        if not self.client:
            return "" # empty when web not configured
        return self.client.run(query, k=k)

    def _truncate_query(self, text: str, limit: int = 300) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def multi_hop(self, query: str, hops: int = 2, per_hop: int = 3) -> str:
    # simple iterative expansion: for more advanced, use LLM to extract intermediate queries
        out_texts = []
        cur = query
        for _ in range(hops):
            res = self.search(cur, k=per_hop)
            out_texts.append(res)
            # use a heuristic: take the first sentence as a follow-up query (can be improved with an LLM)
            if res:
                first_line = res.split('\n')[0]
                cur = self._truncate_query(f"{query} {first_line}", 300)
        return "\n\n".join(out_texts)