from langgraph.graph import StateGraph,START,END
from nodes import (
     load_document, 
     create_database,
     retriever,
     check_retrieved_chunks,
     refine_docs,
     tool_node,
     search_web,
     refine_web_results,
     generate_response_using_docs,
     generate_response_using_both
)
from state import state

graph=StateGraph(state)

def check_ambiguity(state:state):
     if state["status"]=="ambigious": return "ambigious"
     else: return "incorrect"

def quality_check_router(state:state):
     if state["status"]=="correct": return "correct"
     elif state["status"]=="ambigious": return "ambigious"
     else: return "incorrect"

def choose_generator(state:state):
     if state["status"]=="correct": return "response_generator_from_docs"
     elif state["status"]=="ambigious": return "response_generator_from_both"

graph.add_node("load_document",load_document)
graph.add_node("vector_database",create_database)
graph.add_node("retriever",retriever)
graph.add_node("docs_evaluator",check_retrieved_chunks)
graph.add_node("refine_docs",refine_docs)
graph.add_node("tavily_tool",tool_node)
graph.add_node("web_search",search_web)
graph.add_node("refine_web_results",refine_web_results)
graph.add_node("response_generator_from_docs",generate_response_using_docs)
graph.add_node("response_generator_from_both",generate_response_using_both)


graph.add_edge(START,"load_document")
graph.add_edge("load_document","vector_database")
graph.add_edge("vector_database","retriever")
graph.add_edge("retriever","docs_evaluator")
graph.add_conditional_edges("docs_evaluator",quality_check_router,{
     "correct":"refine_docs",
     "ambigious":"web_search",
     "incorrect":"web_search"
})
#for correct case
graph.add_conditional_edges("refine_docs",choose_generator,{
     "response_generator_from_docs":"response_generator_from_docs",
     "response_generator_from_both":"response_generator_from_both"

})
#for ambigious case
graph.add_edge("web_search","tavily_tool")
graph.add_edge("tavily_tool","refine_web_results")
graph.add_conditional_edges("refine_web_results",check_ambiguity,{
     "ambigious":"refine_docs",
     "incorrect":"response_generator_from_docs"
})

graph.add_edge("response_generator_from_docs",END)
graph.add_edge("response_generator_from_both",END)

workflow=graph.compile()