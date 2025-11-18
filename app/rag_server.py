# # app/rag_server.py

# import os
# from flask import Flask, request, jsonify
# from dotenv import load_dotenv
# from psutil import Process

# from pinecone import Pinecone
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore

# from app.agent_builder import build_agent, build_groq_llm

# load_dotenv()

# app = Flask(__name__)

# _agent = None
# _retriever = None


# def mem_mb():
#     return round(Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)


# def init_system():
#     global _agent, _retriever
#     if _agent:
#         return

#     print("🚀 Initializing system…", mem_mb(), "MB")

#     INDEX = os.getenv("PINECONE_INDEX", "medical-chatbot").lower()
#     API_KEY = os.getenv("PINECONE_API_KEY")

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     # Pinecone v4
#     pc = Pinecone(api_key=API_KEY)

#     if INDEX not in pc.list_indexes().names():
#         raise RuntimeError(f"Index '{INDEX}' not found.")

#     vectorstore = PineconeVectorStore.from_existing_index(
#         embedding=embeddings,
#         index_name=INDEX
#     )

#     _retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 3}
#     )

#     llm = build_groq_llm()
#     _agent = build_agent(llm, _retriever)

#     print("✅ Agentic RAG Online — Memory:", mem_mb(), "MB")

# @app.route("/")
# def home():
#     return "Agentic RAG API is running"

# @app.route("/health")
# def health():
#     return {"status": "ok"}


# @app.route("/ask", methods=["POST"])
# def ask():
#     global _agent

#     body = request.json or {}
#     question = body.get("question")

#     if not question:
#         return jsonify({"error": "Missing 'question'"}), 400

#     if not _agent:
#         init_system()

#     try:
#         result = _agent.invoke({"input": question})
#         return jsonify({"answer": result["output"]})
#     except Exception as e:
#         print("\n\n🔥 INTERNAL ERROR:\n", e, "\n\n")   # 👈 ADD THIS
#         raise                                         # 👈 ADD THIS (forces Flask to show traceback)


# if __name__ == "__main__":
#     init_system()
#     app.run(host="0.0.0.0", port=8080 ,debug=True)







# app/rag_server.py

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from psutil import Process

from app.agent_executor import AgentExecutor

load_dotenv()
app = Flask(__name__)

agent = None


def mem_mb():
    return round(Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)


def init_system():
    global agent
    if agent:
        return

    print("🚀 Initializing Level-3 RAG…", mem_mb(), "MB")

    pc_key = os.getenv("PINECONE_API_KEY")
    index = os.getenv("PINECONE_INDEX", "medical-chatbot")
    tavily = os.getenv("TAVILY_API_KEY")

    agent = AgentExecutor(
        pinecone_api=pc_key,
        pinecone_index=index,
        tavily_key=tavily
    )

    print("✅ Level-3 RAG Online — Memory:", mem_mb(), "MB")


@app.route("/ask", methods=["POST"])
def ask():
    global agent
    if not agent:
        init_system()

    question = (request.json or {}).get("question")
    if not question:
        return {"error": "Missing question"}, 400

    return agent.answer(question)


if __name__ == "__main__":
    init_system()
    app.run(host="0.0.0.0", port=8080, debug=True)
