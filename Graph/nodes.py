from LLM import get_llm,get_llm_with_tool,tools
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader,UnstructuredPowerPointLoader
from RAG_pipelines.document_loader import load_file
from RAG_pipelines.embeddings import create_embeddings,load_embeedings
from RAG_pipelines.retriever import retrieve_embeddings
from state import state
import os
import json
import ast
import re
from typing import List
from langgraph.prebuilt import ToolNode

def load_document(state:state):
    path=state['doc_path']
    chunks=load_file(path)
    return{"chunks":chunks}

def create_database(state:state):
    # chunks=state["chunks"]
    database=load_embeedings()
    return{"database":database}

def retriever(state:state):
    db=state["database"]
    query=state["query"]
    retrieved_docs=retrieve_embeddings({"query":query , "db":db})
    return {"retrieved_docs":retrieved_docs}

def check_retrieved_chunks(state:state):
    print("\n Evaluating retrieved chunks \n")
    query=state["query"]
    context=state["retrieved_docs"]
    upper_threshold=0.7
    lower_threshold=0.3
    scores=[]
    good_docs=[]
    status="not assigned"
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a retrieval evaluator in a Corrective Retrieval-Augmented Generation (CRAG) system.

        Your task is to evaluate how relevant a retrieved document chunk is to the user's query.

        Evaluate the document based ONLY on its ability to help answer the query.

        Scoring criteria:
        - 0.00–0.29: Irrelevant. The document does not contain useful information for the query.
        - 0.30–0.59: Weakly relevant. The document has some related information but is insufficient or mostly indirect.
        - 0.60–0.79: Relevant. The document contains useful information that can help answer the query.
        - 0.80–1.00: Highly relevant. The document directly contains important information needed to answer the query.

        Consider semantic relevance, factual usefulness, and how directly the document addresses the query.
        Do NOT judge writing quality, grammar, or whether the document completely answers the query.

        IMPORTANT:
        Return ONLY a single floating-point number between 0.0 and 1.0.
        Do not return words, explanations, labels, JSON, markdown, or any other text.
        """
    ),
    (
        "human",
        """
        User Query:
        {query}

        Retrieved Document:
        {document}

        Relevance Score:
        """
    )
    ])
    chain=RunnableSequence(prompt | get_llm() )
    for doc in context:
        response=chain.invoke({"document":doc.page_content , "query":query})
        if float(response.content.strip()) >= lower_threshold:
            good_docs.append(doc)
            scores.append(float(response.content.strip()))
    if scores:
        for score in scores:
            if score >= upper_threshold:
                status="correct"
                break
            elif score >= lower_threshold: status="ambigious"
            else: status="incorrect"
    else: status="incorrect"

    return {"good_docs":good_docs , "scores":scores , "status":status}

class refinement_tools:
    @staticmethod
    def decompose_to_sentences(text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a strict relevance evaluator in a Corrective Retrieval-Augmented Generation (CRAG) system.

        Your task is to determine whether the retrieved sentence contains information that is relevant and useful for answering the user's query.

        Return True ONLY when the sentence provides information that directly or meaningfully helps answer the query.

        Return False when:
        - The sentence is unrelated to the query.
        - The sentence only shares a weak or superficial keyword overlap.
        - The sentence provides background that does not help answer the query.
        - The sentence is ambiguous or too vague to be useful.
        - The sentence contradicts or is unrelated to what the query is asking.

        Do not infer information that is not explicitly present in the sentence.
        Do not use outside knowledge.
        Judge the sentence only in relation to the given query.

        IMPORTANT:
        Your response MUST contain exactly one of these two values:

        True
        False

        Do not return explanations, punctuation, quotes, JSON, markdown, or any other text.
        """
    ),
    (
        "human",
        """
        Query:
        {query}

        Sentence:
        {sentence}

        Is this sentence relevant and useful for answering the query?
        """
    )
    ])

    kept_strips:list=[]
    chain=RunnableSequence(prompt | get_llm() )

    def get_refined_content(self,query:str , context:str):
        sentences=self.decompose_to_sentences(context)
        for sentence in sentences:
            response=self.chain.invoke({"query": query, "sentence": sentence})
            if response.content.strip() == "True": self.kept_strips.append(sentence)

        return "".join(strip for strip in self.kept_strips).strip()

def refine_docs(state:state):
    print("\n refining docs \n")
    obj=refinement_tools()

    context="".join(doc.page_content for doc in state["good_docs"]).strip()

    refined_content=obj.get_refined_content(state["query"],context)

    return {"refined_docs_context":refined_content}


tool_node=ToolNode(tools)

def search_web(state:state):
    print("\n searching Web \n")
    query=state["query"]
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a web-search agent in a Corrective Retrieval-Augmented Generation (CRAG) system.

        Your task is to search the web for the user's query using the available Tavily search tool.

        Instructions:
        - Use the Tavily search tool to find reliable and relevant information for the query.
        - Search for the query exactly as needed to retrieve useful web sources.
        - Prefer authoritative, trustworthy, and directly relevant sources.
        - Retrieve up to 3 search results.
        - Do not answer the user's query yourself.
        - Do not summarize, interpret, or rewrite the search results.
        - Your job is only to perform the web search and return the tool results.
        """
    ),
    (
        "human",
        """
        Search the web for the following query:

        {query}
        """
    )
    ])

    chain=RunnableSequence(prompt | get_llm_with_tool())
    response=chain.invoke({
        "query":query
    })

    return {"messages":[response]}




