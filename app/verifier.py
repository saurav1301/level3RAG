# app/verifier.py
from typing import List
from langchain_core.documents import Document


# Very simple evidence checker: check that final answer strings contain facts present in retrieved snippets.


def verify(answer: str, evidence: List[Document], threshold: float = 0.1) -> dict:
# returns: {ok: bool, reasons: [..], matched: [...]}
    matched = []
    for d in evidence:
        snippet = d.page_content
        # cheap substring check; replace with semantic matching for more robust verification
        if len(snippet) > 40 and any(chunk.strip() in answer for chunk in snippet.split('\n')[:3]):
            matched.append(d)
    ok = len(matched) > 0
    return {"ok": ok, "matched": matched, "count": len(matched)}