from typing import TypedDict,List,Dict,Annotated
from langgraph.graph import add_messages
from langchain_core.documents import Document

class state(TypedDict):
     query:str
     chunks:List[Document]
     retrieved_docs:List[Document]
     response:str
