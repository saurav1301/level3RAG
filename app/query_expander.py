# app/query_expander.py
from typing import List


# Simple template-based expander. Replace with LLM-based paraphraser if needed.
TEMPLATES = [
    "{q}",
    "What are the latest updates about {q}",
    "Explain {q} in simple terms",
    "Recent news about {q}",
    "{q} research summary",
]




def expand_query(query: str) -> List[str]:
    query = query.strip()
    out = []
    for t in TEMPLATES:
        out.append(t.format(q=query))
    # always include original
    if query not in out:
        out.insert(0, query)
    return out