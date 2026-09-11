from LLM import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader,UnstructuredPowerPointLoader
from RAG_pipelines.document_loader import load_file
from RAG_pipelines.embeddings import create_embeddings,load_embeedings
from RAG_pipelines.retriever import retrieve_embeddings
from state import state
import os

def load_document(state:state):
    path=state['doc_path']
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".pdf":
        docs= PyPDFLoader(path)
        text=docs.load()
    elif ext == ".csv":
        docs=CSVLoader(file_path=path, encoding="utf-8")
        text=doc.load()
    elif ext in [".ppt", ".pptx"]:
        docs=UnstructuredPowerPointLoader(path)
        text=doc.load()
    elif ext == ".txt":
        docs=TextLoader(path)
        text=doc.load()
    else:
        raise ValueError("Unsupported file type")

    return{""}

