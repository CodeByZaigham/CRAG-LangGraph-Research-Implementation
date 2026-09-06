from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

EMBEDDING_MODEL=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

@traceable(name="creating vector embeddings")
def create_embeddings(chunks:list):
        database=FAISS.from_documents(chunks,EMBEDDING_MODEL)
        return database

@traceable(name="loading vector embeddings")
def load_embeedings():
        pass