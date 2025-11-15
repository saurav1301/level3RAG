# scripts/build_pinecone_index.py
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Pinecone v4 imports
from pinecone import Pinecone, ServerlessSpec

# Pinecone + LangChain wrapper
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "medical-chatbot").lower()

if not PINECONE_API_KEY:
    raise SystemExit("Please set PINECONE_API_KEY in .env")


# ---------------------------
# 1) Load PDFs
# ---------------------------
print("Loading PDFs...")
loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
print(f"Loaded {len(docs)} documents.")

# ---------------------------
# 2) Chunking
# ---------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")

# ---------------------------
# 3) Embeddings
# ---------------------------
print("Initializing HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# 4) Pinecone v4 Init
# ---------------------------
print("Connecting to Pinecone (v4)...")
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [i["name"] for i in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# connect to index
index = pc.Index(INDEX_NAME)


# ---------------------------
# 5) Upload vectors
# ---------------------------
print("Uploading embeddings to Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("🎉 DONE! Pinecone index built successfully.")
