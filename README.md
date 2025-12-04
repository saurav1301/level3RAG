<<<<<<< HEAD
✨ Features (What makes this Level-3 RAG?)
🧩 1. Multi-Query Retrieval

Automatically expands the user query into 5+ semantic variants to maximize recall.
This avoids “query miss” problems in small datasets.

🔥 2. Cross-Encoder Reranking

Uses Sentence-Transformers similarity scoring to produce a high-precision final ranking.

This fixes:

Wrong document picked due to vector noise

Low-relevance answers

“Nearest chunk but wrong meaning” RAG failures

🌐 3. Real-Time Web Search (Tavily API)

The system can fetch live, real-time information from the public internet.

Supports:

Multi-hop reasoning

Multi-source aggregation

Evidence fusion

🔗 4. Hybrid Fusion Engine

Both local medical PDF knowledge and web evidence are combined into the final answer.

Result = more accurate + up-to-date.

🛡 5. Verification Engine

After generating an answer, the model checks whether each claim is supported by local evidence.

Detects:

Hallucinations

Unsupported statements

Weak citations

🧠 6. Groq Llama-3.1 — Ultra Fast LLM

Backed by Groq’s blazing fast AI accelerators.
latency ~5–20ms per token.

📦 7. Pinecone Vector Database (Serverless v4)

Stores chunked embeddings from medical PDFs.

⚙ 8. Flask API — Ready for Integration

POST /ask → returns structured JSON:

{
  "answer": "...",
  "local_used": 5,
  "verification": { "ok": true, "count": 4, "matched": [...] }
}

🧱 High-Level Architecture
                    ┌─────────────────────────────┐
                    │  User Question               │
                    └──────────────┬──────────────┘
                                   │
                          Multi-Query Expansion
                                   │
                    ┌──────────────▼──────────────┐
                    │    Pinecone Vector DB        │
                    │ (Local Knowledge Retrieval)  │
                    └──────────────┬──────────────┘
                                   │ Retrieved Chunks
                              Reranker (Cross Encoder)
                                   │
                 ┌─────────────────┴───────────────────┐
                 │     Multi-Hop Web Search (Tavily)   │
                 └─────────────────┬───────────────────┘
                                   │
                           Evidence Fusion
                                   │
                              Groq Llama 3.1
                                   │
                             Final Answer
                                   │
                           Verification Engine
                                   │
                         JSON Output via Flask

📂 Directory Structure
agentic_RAG/
│
├── app/
│   ├── agent_builder.py        # Old LCEL agent (optional)
│   ├── agent_executor.py       # Main Level-3 pipeline
│   ├── multi_retriever.py      # Multi-query retriever
│   ├── reranker.py             # Cross-encoder reranking
│   ├── query_expander.py       # Query paraphrasing
│   ├── web_pipeline.py         # Multi-hop web search
│   ├── tools.py                # Pinecone + Tavily + helpers
│   ├── verifier.py             # Consistency checker
│   ├── rag_server.py           # Flask API
│   └── __init__.py
│
├── scripts/
│   └── build_pinecone_index.py # One-time indexing script
│
├── data/
│   └── Medical_book.pdf        # Local knowledge source
│
└── docker/
    ├── Dockerfile
    └── start.sh

🛠 Setup Instructions
1️⃣ Install Requirements
pip install -r requirements.txt

2️⃣ Set Environment Variables

Create .env:

PINECONE_API_KEY=your_key
PINECONE_INDEX=medical-chatbot
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

3️⃣ Build Pinecone Index (only once)
python scripts/build_pinecone_index.py

4️⃣ Run API Server
python -m app.rag_server

🧪 API Usage (POST Request)
POST /ask

Request:

{
  "question": "What are the symptoms of dengue?"
}


Response:

{
  "answer": "Dengue symptoms include...",
  "local_used": 5,
  "verification": {
    "ok": false,
    "count": 0,
    "matched": []
  }
}

🧪 Test Cases
✔ Local-only question

"Explain the causes of anemia."

✔ Web-only live info

"What are the latest WHO guidelines for dengue in 2025?"

✔ Local + Web fusion

"Compare local vs latest updates for dengue symptoms."

✔ Math + Logic Tooling

"A patient takes 250mg medicine 3 times daily. How much per week?"

✔ Long-context stress test

"Summarize the entire dengue section."

✔ Verification test

"Does the evidence support: dengue causes purple fingers?"

🧠 Why This Project Is Special (Recruiter Pitch)

This is not a basic chatbot.
It is a full AI retrieval system with:

Dynamic retrieval strategies

Multi-hop search

Evidence-based reasoning

Grounded outputs

Industry architecture (Perplexity-style)

Real-time web sourcing

LLM + vector + reranking synergy

Recruiters see:

LLM Ops

Production AI engineering

Retrieval pipelines

API design

Search engineering

Embedding models

Pinecone expertise

Groq inference

Web search agents

You look like someone who can design & deploy scalable AI systems, not just toy apps.

That’s how 20 LPA happens. 😉

📜 License

MIT License.

⭐ If you like this project…

Leave a ⭐ on the repo and connect on LinkedIn.
=======
# level3RAG
>>>>>>> 6106c705339d1def4dad91c189bfb049839a080e
