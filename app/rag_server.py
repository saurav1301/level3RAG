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
