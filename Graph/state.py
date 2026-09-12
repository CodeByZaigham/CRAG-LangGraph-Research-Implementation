from typing import TypedDict,List,Dict,Annotated,Literal
from langgraph.graph import add_messages
from langchain_core.documents import Document
from langchain_chroma import Chroma

class state(TypedDict):
     # traditional RAG 

     doc_path:str
     query:str
     chunks:List[Document]
     database:Chroma
     retrieved_docs:List[Document]
     response:str

     #for evaluating docs

     good_docs:List[Document]
     scores:List[float]
     status:Literal["correct" , "incorrect" , "ambigious"]

     #for refining docs

     kept_strips:List[str]
     refined_context:str
