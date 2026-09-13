from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from pydantic import BaseModel, Field

_tavily = TavilySearch(max_results=3)

class TavilyInput(BaseModel):
    query: str = Field(description="The search query to look up")

@tool("tavily_search", args_schema=TavilyInput)
def web_search(query: str) -> str:
    """Search the web for current information on a topic."""
    # Only 'query' is ever passed through — max_results is baked in above
    return _tavily.invoke({"query": query})



tools=[web_search]

def get_llm():
     return ChatGroq(model="openai/gpt-oss-120b")

def get_llm_with_tool():
     llm=ChatGroq(model="openai/gpt-oss-120b")
     return llm.bind_tools(tools)
     
