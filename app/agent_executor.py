# # app/agent_executor.py
# import os
# from typing import List
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate


# from app.multi_retriever import MultiQueryRetriever
# from app.reranker import rerank
# from app.web_pipeline import WebPipeline
# from app.verifier import verify
# from app.tools import init_pinecone_index




# class AgentExecutor:
#     def __init__(self, pinecone_api, pinecone_index, tavily_key=None):
#         emb_model = os.getenv('HF_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
#         self.vectorstore = init_pinecone_index(pinecone_api, pinecone_index, emb_model)
#         self.retriever = MultiQueryRetriever(self.vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 8}))
#         self.web = WebPipeline(tavily_key)
#         api_key = os.getenv('GROQ_API_KEY')
#         if not api_key:
#             raise RuntimeError('GROQ_API_KEY not set')
#         self.llm = ChatGroq(model='llama-3.1-8b-instant', api_key=api_key, temperature=0.0)


#             # Prompt template
#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are an assistant that cites sources and lists provenance. Answer concisely."),
#             ("human", "{input}")
#         ])


#     def answer(self, question: str) -> dict:
#     # 1) Retrieve local docs
#         docs = self.retriever.retrieve(question, top_k=8)
#         # 2) Rerank
#         ranked = rerank(question, docs, top_k=5)
#         # 3) Web search multi-hop
#         web = self.web.multi_hop(question, hops=2, per_hop=3)
#         # 4) Compose prompt with evidence
#         evidence_text = "\n\n".join([d.page_content[:800] for d in ranked])
#         prompt_in = f"Question:\n{question}\n\nLocal Evidence:\n{evidence_text}\n\nWeb Evidence:\n{web}\n\nAnswer briefly and cite which evidence you used (Local #1..N or Web)."
#         msgs = self.prompt.format_messages(input=prompt_in)
#         resp = self.llm.invoke(msgs)
#         answer = resp.content.strip()
#         # 5) Verification
#         verification = verify(answer, ranked)
#         return {
#             'answer': answer,
#             'local_count': len(ranked),
#             'verification': verification
#         }






# app/agent_executor.py

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.tools import init_pinecone_index
from app.multi_retriever import MultiQueryRetriever
from app.reranker import rerank
from app.web_pipeline import WebPipeline
from app.verifier import verify


class AgentExecutor:
    def __init__(self, pinecone_api, pinecone_index, tavily_key=None):

        # 1. Pinecone init
        emb_model = os.getenv('HF_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.vectorstore = init_pinecone_index(
            pinecone_api,
            pinecone_index,
            emb_model
        )

        # 2. MultiQuery Retrieval (base retriever)
        base = self.vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 8}
        )
        self.retriever = MultiQueryRetriever(base)

        # 3. Web pipeline
        self.web = WebPipeline(tavily_key)

        # 4. LLM
        api_key = os.getenv('GROQ_API_KEY')
        self.llm = ChatGroq(
            model='llama-3.1-8b-instant',
            api_key=api_key,
            temperature=0.0
        )

        # 5. Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that answers using local+web evidence. Cite sources."),
            ("human", "{input}")
        ])


    def answer(self, question: str) -> dict:

        # 1) Retrieve
        docs = self.retriever.retrieve(question, top_k=8)

        # 2) Rerank
        ranked = rerank(question, docs, top_k=5)

        # 3) Web multi-hop
        web = self.web.multi_hop(question, hops=2, per_hop=3)

        # 4) Compose evidence
        evidence_text = "\n\n".join(
            [d.page_content[:800] for d in ranked]
        )

        prompt_input = f"""
QUESTION:
{question}

LOCAL EVIDENCE:
{evidence_text}

WEB EVIDENCE:
{web}

Answer concisely. Cite sources as (Local #1..#N) or (Web).
"""

        msgs = self.prompt.format_messages(input=prompt_input)
        resp = self.llm.invoke(msgs)
        answer = resp.content.strip()

        # 5) Verify hallucinations
        verification = verify(answer, ranked)

        return {
            "answer": answer,
            "local_used": len(ranked),
            "verification": verification
        }
