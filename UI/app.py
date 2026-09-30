"""
CRAG (Corrective Retrieval-Augmented Generation) - Research Implementation UI
Conference Paper Implementation Interactive Interface
Paper: "Corrective Retrieval-Augmented Generation" (Yan et al., 2024 - arXiv:2401.15884)
Orchestration: LangGraph | Embeddings: HuggingFace MiniLM-L6-v2 | VectorDB: Chroma | Web Search: Tavily
"""

import os
import sys
import json
import time
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. Environment & Path Setup
# -----------------------------------------------------------------------------
load_dotenv()

UI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(UI_DIR)
GRAPH_DIR = os.path.join(PROJECT_ROOT, "Graph")
PDFS_DIR = os.path.join(PROJECT_ROOT, "Pdfs")

if GRAPH_DIR not in sys.path:
    sys.path.insert(0, GRAPH_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------------------------------------------------------
# 2. Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CRAG | Corrective RAG Research Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 3. Custom CSS for Academic / Modern Research Aesthetic
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Theme Font & Global Reset */
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .stCodeBlock {
        font-family: 'Fira Code', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 95% !important;
    }

    /* Gradient Hero Card */
    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.98rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 8px;
        margin-top: 10px;
    }
    .badge-blue { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-green { background: rgba(34, 197, 94, 0.15); color: #4ADE80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-red { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Metric & Action Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 16px 20px;
        transition: all 0.2s ease-in-out;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
    }

    /* Response Container */
    .response-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 14px;
        padding: 24px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(56, 189, 248, 0.15);
    }

    /* Strip / Sentence Highlight Card */
    .strip-item {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #38BDF8;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #E2E8F0;
        line-height: 1.5;
    }
    .strip-web {
        border-left-color: #C084FC;
    }

    /* Score Indicator Gauge */
    .score-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .score-high { background: rgba(34, 197, 94, 0.2); color: #4ADE80; border: 1px solid #22C55E; }
    .score-mid { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }
    .score-low { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444; }

    /* Custom Button Styling */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 4. Pipeline Imports with Graceful Fallback
# -----------------------------------------------------------------------------
workflow_import_error = None
workflow = None
load_file_func = None
create_embeddings_func = None
load_embeddings_func = None

try:
    from Graph.edges import workflow
    from Graph.RAG_pipelines.document_loader import load_file as load_file_func
    from Graph.RAG_pipelines.embeddings import create_embeddings as create_embeddings_func, load_embeedings as load_embeddings_func
except Exception as e:
    workflow_import_error = traceback.format_exc()

# -----------------------------------------------------------------------------
# 5. Session State Initialization
# -----------------------------------------------------------------------------
if "doc_path" not in st.session_state:
    default_pdf = os.path.join(PDFS_DIR, "deep learning book.pdf")
    st.session_state.doc_path = default_pdf if os.path.exists(default_pdf) else ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_run" not in st.session_state:
    st.session_state.latest_run = None

if "preset_query" not in st.session_state:
    st.session_state.preset_query = ""

# -----------------------------------------------------------------------------
# 6. Graph Flow Metadata & Visual Graphviz Renderer
# -----------------------------------------------------------------------------
NODE_METADATA = {
    "load_document": {
        "icon": "📄",
        "title": "Document Loader",
        "desc": "Loads & partitions document into semantic chunks",
    },
    "vector_database": {
        "icon": "🧱",
        "title": "Vector Database",
        "desc": "Initializes Chroma vector store with MiniLM embeddings",
    },
    "retriever": {
        "icon": "🔍",
        "title": "Multi-Query Retriever",
        "desc": "Generates query variations & extracts top candidate chunks",
    },
    "docs_evaluator": {
        "icon": "⚖️",
        "title": "Retrieval Evaluator",
        "desc": "Evaluates semantic chunk relevance & confidence scores",
    },
    "refine_docs": {
        "icon": "✂️",
        "title": "Doc Knowledge Refiner",
        "desc": "Sentence-level decomposition & strict relevance strip filtering",
    },
    "web_search": {
        "icon": "🌐",
        "title": "Web Search Query Agent",
        "desc": "Synthesizes external search query for Tavily tool",
    },
    "tavily_tool": {
        "icon": "📡",
        "title": "Tavily Search Engine",
        "desc": "Executes real-time web retrieval for supplementary facts",
    },
    "refine_web_results": {
        "icon": "🔬",
        "title": "Web Knowledge Refiner",
        "desc": "Filters & cleans external web snippets into verified knowledge strips",
    },
    "response_generator_from_docs": {
        "icon": "✍️",
        "title": "Grounded Generator (Docs/Web)",
        "desc": "Synthesizes final answer grounded strictly in refined context",
    },
    "response_generator_from_both": {
        "icon": "🔀",
        "title": "Hybrid-Grounded Generator",
        "desc": "Merges verified internal document strips with web knowledge strips",
    },
}

def generate_crag_graphviz(
    current_node: Optional[str] = None,
    visited_nodes: Optional[List[str]] = None,
    status_state: Optional[str] = None,
) -> str:
    """
    Renders an academic-grade Graphviz DOT flow representation of the CRAG LangGraph pipeline.
    Nodes glow dynamically based on execution state (active, completed, branch taken).
    """
    visited = set(visited_nodes or [])

    # Color Palette
    BG_COLOR = "#0B0F19"
    NODE_DEFAULT_BG = "#1E293B"
    NODE_DEFAULT_BORDER = "#475569"
    NODE_DEFAULT_FONT = "#94A3B8"
    
    NODE_ACTIVE_BG = "#0284C7"
    NODE_ACTIVE_BORDER = "#38BDF8"
    NODE_ACTIVE_FONT = "#FFFFFF"

    NODE_DONE_BG = "#064E3B"
    NODE_DONE_BORDER = "#10B981"
    NODE_DONE_FONT = "#ECFDF5"

    EDGE_COLOR = "#64748B"
    EDGE_ACTIVE_COLOR = "#38BDF8"
    EDGE_CORRECT_COLOR = "#10B981"
    EDGE_AMBIGUOUS_COLOR = "#F59E0B"
    EDGE_INCORRECT_COLOR = "#EF4444"

    def get_node_style(node_id: str, label_text: str) -> str:
        if node_id == current_node:
            bg, border, font = NODE_ACTIVE_BG, NODE_ACTIVE_BORDER, NODE_ACTIVE_FONT
            penwidth = "2.5"
        elif node_id in visited:
            bg, border, font = NODE_DONE_BG, NODE_DONE_BORDER, NODE_DONE_FONT
            penwidth = "1.8"
        else:
            bg, border, font = NODE_DEFAULT_BG, NODE_DEFAULT_BORDER, NODE_DEFAULT_FONT
            penwidth = "1.0"
        return f'{node_id} [label="{label_text}", fillcolor="{bg}", color="{border}", fontcolor="{font}", penwidth={penwidth}];'

    dot = [
        "digraph CRAG {",
        '  graph [rankdir=TB, bgcolor="transparent", margin="0.1", nodesep="0.45", ranksep="0.55"];',
        '  node [shape=box, style="rounded,filled", fontname="Inter, Helvetica", fontsize=10, margin="0.2,0.12"];',
        '  edge [fontname="Inter, Helvetica", fontsize=8, color="#64748B", fontcolor="#94A3B8", arrowsize=0.7];',
        '  START [shape=circle, label="START", width=0.5, style=filled, fillcolor="#334155", color="#64748B", fontcolor="#F8FAFC", fontsize=8];',
        '  END [shape=doublecircle, label="END", width=0.5, style=filled, fillcolor="#334155", color="#64748B", fontcolor="#F8FAFC", fontsize=8];',
    ]

    # Node definitions
    dot.append(get_node_style("load_document", "📄 Load Document"))
    dot.append(get_node_style("vector_database", "🧱 Vector DB (Chroma)"))
    dot.append(get_node_style("retriever", "🔍 Multi-Query Retriever"))
    dot.append(get_node_style("docs_evaluator", "⚖️ Evaluator & Confidence Router"))
    dot.append(get_node_style("refine_docs", "✂️ Refine Doc Context"))
    dot.append(get_node_style("web_search", "🌐 Web Search Planner"))
    dot.append(get_node_style("tavily_tool", "📡 Tavily Tool Node"))
    dot.append(get_node_style("refine_web_results", "🔬 Refine Web Context"))
    dot.append(get_node_style("response_generator_from_docs", "✍️ Doc/Web Grounded Gen"))
    dot.append(get_node_style("response_generator_from_both", "🔀 Hybrid Grounded Gen"))

    # Edges
    dot.append('  START -> load_document;')
    dot.append('  load_document -> vector_database;')
    dot.append('  vector_database -> retriever;')
    dot.append('  retriever -> docs_evaluator;')

    # Conditional router branches from docs_evaluator
    eval_is_done = "docs_evaluator" in visited
    c_color = EDGE_CORRECT_COLOR if (eval_is_done and status_state == "correct") else EDGE_COLOR
    a_color = EDGE_AMBIGUOUS_COLOR if (eval_is_done and status_state == "ambigious") else EDGE_COLOR
    i_color = EDGE_INCORRECT_COLOR if (eval_is_done and status_state == "incorrect") else EDGE_COLOR

    dot.append(f'  docs_evaluator -> refine_docs [label="[Correct: >= 0.7]", color="{c_color}", fontcolor="{c_color}", penwidth=1.5];')
    dot.append(f'  docs_evaluator -> web_search [label="[Ambiguous: 0.3-0.7]", color="{a_color}", fontcolor="{a_color}", penwidth=1.5];')
    dot.append(f'  docs_evaluator -> web_search [label="[Incorrect: < 0.3]", color="{i_color}", fontcolor="{i_color}", penwidth=1.5];')

    # Refine docs branches
    dot.append('  refine_docs -> response_generator_from_docs [label="Correct Path"];')
    dot.append('  refine_docs -> response_generator_from_both [label="Ambiguous Path"];')

    # Web search flow
    dot.append('  web_search -> tavily_tool;')
    dot.append('  tavily_tool -> refine_web_results;')
    dot.append(f'  refine_web_results -> refine_docs [label="Ambiguous -> Merge Docs", color="{a_color}"];')
    dot.append(f'  refine_web_results -> response_generator_from_docs [label="Incorrect -> Web Only", color="{i_color}"];')

    # End connections
    dot.append('  response_generator_from_docs -> END;')
    dot.append('  response_generator_from_both -> END;')

    dot.append("}")
    return "\n".join(dot)

# -----------------------------------------------------------------------------
# 7. Helper Functions: Document Ingestion & Execution
# -----------------------------------------------------------------------------
def save_uploaded_file(uploaded_file) -> str:
    """Saves an uploaded Streamlit file to a temporary workspace directory."""
    temp_dir = os.path.join(PROJECT_ROOT, "scratch_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def build_vector_index(file_path: str):
    """Executes document loading, text splitting, and vector embedding creation."""
    if not load_file_func or not create_embeddings_func:
        st.error("RAG pipeline modules are not available for indexing.")
        return False
    try:
        with st.spinner("Partitioning document into semantic chunks..."):
            chunks = load_file_func(file_path)
        with st.spinner(f"Generating embeddings for {len(chunks)} chunks & populating ChromaDB..."):
            create_embeddings_func(chunks)
        return len(chunks)
    except Exception as e:
        st.error(f"Error during vector index construction: {str(e)}")
        st.code(traceback.format_exc())
        return False

def format_node_output_summary(node_name: str, output: Any) -> str:
    """Generates a concise, informative execution summary for each LangGraph node."""
    if not isinstance(output, dict):
        return ""
    try:
        if node_name == "load_document":
            chunks = output.get("chunks", [])
            return f"Loaded & split into **{len(chunks)}** chunk(s)"
        elif node_name == "vector_database":
            return "Chroma vector database loaded successfully"
        elif node_name == "retriever":
            docs = output.get("retrieved_docs", [])
            return f"Retrieved **{len(docs)}** candidate chunk(s) via multi-query expansion"
        elif node_name == "docs_evaluator":
            status = output.get("status", "unknown").upper()
            scores = output.get("scores", [])
            good_docs = output.get("good_docs", [])
            avg_score = f"{sum(scores)/len(scores):.2f}" if scores else "0.00"
            return f"Evaluation: **{status}** | Evaluated: **{len(scores)}** | High-Confidence Chunks: **{len(good_docs)}** | Avg Score: **{avg_score}**"
        elif node_name == "refine_docs":
            context = output.get("refined_docs_context", "")
            return f"Refined document context: **{len(context.split())}** words kept after sentence filtering"
        elif node_name == "web_search":
            return "Generated Tavily search tool-call dispatch"
        elif node_name == "tavily_tool":
            return "Tavily web retrieval executed and search results returned"
        elif node_name == "refine_web_results":
            context = output.get("refined_web_context", "")
            return f"Refined web context: **{len(context.split())}** words extracted from web sources"
        elif node_name in ("response_generator_from_docs", "response_generator_from_both"):
            resp = output.get("response", "")
            return f"Final synthesized grounded response: **{len(resp.split())}** words"
    except Exception:
        return ""
    return ""

def execute_crag_pipeline(query: str, doc_path: str):
    """
    Generator streaming LangGraph execution steps, yielding node names, outputs,
    and cumulative state snapshots for real-time UI animation.
    """
    initial_state = {
        "doc_path": doc_path or "",
        "query": query,
    }
    full_state = dict(initial_state)

    if workflow is None:
        raise RuntimeError("CRAG workflow is not initialized or failed to import.")

    for update in workflow.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in update.items():
            if isinstance(node_output, dict):
                full_state.update(node_output)
            yield node_name, node_output, dict(full_state)

# -----------------------------------------------------------------------------
# 8. Sidebar Controls & Research Hyperparameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <span style="font-size:2rem;">🔬</span>
            <div>
                <h3 style="margin:0; font-size:1.15rem; font-weight:700;">CRAG Control Hub</h3>
                <span style="font-size:0.75rem; color:#94A3B8;">Research Lab & Diagnostics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Keys Status Monitor
    st.markdown("#### 🔑 API Key Status")
    groq_key = os.getenv("GROQ_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if groq_key:
            st.markdown("🟢 **Groq LLM**<br><span style='font-size:0.72rem; color:#4ADE80;'>Connected</span>", unsafe_allow_html=True)
        else:
            st.markdown("🔴 **Groq LLM**<br><span style='font-size:0.72rem; color:#F87171;'>Missing Key</span>", unsafe_allow_html=True)
    with col_k2:
        if tavily_key:
            st.markdown("🟢 **Tavily Web**<br><span style='font-size:0.72rem; color:#4ADE80;'>Connected</span>", unsafe_allow_html=True)
        else:
            st.markdown("🔴 **Tavily Web**<br><span style='font-size:0.72rem; color:#F87171;'>Missing Key</span>", unsafe_allow_html=True)

    with st.expander("⚙️ Set API Keys Dynamically", expanded=not (groq_key and tavily_key)):
        user_groq = st.text_input("Groq API Key", value=groq_key, type="password")
        user_tavily = st.text_input("Tavily API Key", value=tavily_key, type="password")
        if st.button("Save API Keys", use_container_width=True):
            if user_groq:
                os.environ["GROQ_API_KEY"] = user_groq
            if user_tavily:
                os.environ["TAVILY_API_KEY"] = user_tavily
            st.success("API Keys updated for current session!")
            st.rerun()

    st.markdown("---")

    # Document & Knowledge Base Management
    st.markdown("#### 📚 Document Management")
    
    # Pre-existing PDFs detection
    available_pdfs = []
    if os.path.exists(PDFS_DIR):
        available_pdfs = [f for f in os.listdir(PDFS_DIR) if f.lower().endswith(('.pdf', '.txt', '.csv', '.ppt', '.pptx'))]

    doc_source_option = st.radio(
        "Select Document Source",
        ["Preloaded Research PDF", "Upload Custom Document", "Direct File Path"],
        index=0,
    )

    if doc_source_option == "Preloaded Research PDF":
        if available_pdfs:
            selected_pdf = st.selectbox("Available Documents", available_pdfs)
            st.session_state.doc_path = os.path.join(PDFS_DIR, selected_pdf)
        else:
            st.info("No documents found in `Pdfs/` folder.")
    elif doc_source_option == "Upload Custom Document":
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "txt", "csv", "ppt", "pptx"],
            help="Supports PDF, TXT, CSV, PowerPoint files",
        )
        if uploaded_file is not None:
            saved_path = save_uploaded_file(uploaded_file)
            st.session_state.doc_path = saved_path
            st.success(f"Loaded: `{uploaded_file.name}`")
    else:
        manual_path = st.text_input("Absolute File Path", value=st.session_state.doc_path)
        if manual_path != st.session_state.doc_path:
            st.session_state.doc_path = manual_path

    if st.session_state.doc_path:
        st.caption(f"📁 Active Document: `{os.path.basename(st.session_state.doc_path)}`")

    # ChromaDB Vector Store Actions
    db_path = os.path.join(PROJECT_ROOT, "chromadb")
    db_exists = os.path.exists(db_path)

    col_db1, col_db2 = st.columns(2)
    with col_db1:
        if st.button("⚡ Index / Re-Index", use_container_width=True, help="Extracts chunks and builds vector database"):
            if not st.session_state.doc_path or not os.path.exists(st.session_state.doc_path):
                st.error("Please provide a valid document path first!")
            else:
                chunk_cnt = build_vector_index(st.session_state.doc_path)
                if chunk_cnt:
                    st.success(f"Indexed {chunk_cnt} chunks into ChromaDB!")
                    st.rerun()

    with col_db2:
        if st.button("🗑️ Clear DB", use_container_width=True, disabled=not db_exists, help="Wipes the local ChromaDB index"):
            shutil.rmtree(db_path, ignore_errors=True)
            st.warning("Vector database wiped.")
            st.rerun()

    st.markdown(
        f"<span style='font-size:0.75rem; color:{'#4ADE80' if db_exists else '#94A3B8'};'>"
        f"{'🟢 Chroma Vector DB Ready' if db_exists else '⚪ Vector DB not yet created'}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # CRAG Hyperparameters Information
    st.markdown("#### ⚙️ CRAG Parameters")
    with st.expander("View Threshold Configuration", expanded=False):
        st.markdown(
            """
            - **Upper Confidence Threshold**: `0.70` (Route $\\to$ Document Grounded)
            - **Lower Confidence Threshold**: `0.30` (Route $\\to$ Ambiguous / Web Fallback)
            - **Similarity Multi-Query `k`**: `4` candidate chunks
            - **Tavily Web Search Max Results**: `3` web sources
            - **Sentence Decomposition**: Dynamic regex boundary splitting
            - **Grounding Verification**: Zero-shot strict context constraint
            """
        )

    # Reset & Utilities
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.latest_run = None
        st.rerun()

# -----------------------------------------------------------------------------
# 9. Main Research Header & Hero Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
            <div>
                <div class="hero-title">CRAG: Corrective Retrieval-Augmented Generation</div>
                <div class="hero-subtitle">
                    Adaptive Self-Correcting RAG Architecture with Retrieval Confidence Evaluation, 
                    Sentence-Level Knowledge Refinement, and Dynamic Web Augmentation.
                </div>
            </div>
            <div style="text-align:right;">
                <span class="badge-pill badge-blue">LangGraph</span>
                <span class="badge-pill badge-purple">ChromaDB</span>
                <span class="badge-pill badge-green">Tavily Search</span>
                <span class="badge-pill badge-amber">Groq GPT-OSS</span>
            </div>
        </div>
        <div style="margin-top:14px; display:flex; gap:18px; font-size:0.82rem; color:#94A3B8; flex-wrap:wrap;">
            <span>📄 <b>Paper Reference:</b> <a href="https://arxiv.org/pdf/2401.15884" target="_blank" style="color:#38BDF8; text-decoration:none;">arXiv:2401.15884</a></span>
            <span>🧩 <b>Graph Flow:</b> Evaluator ➔ Conditional Router ➔ Knowledge Strips ➔ Grounded Synthesis</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pipeline Import Error Warning Banner
if workflow_import_error:
    st.error("🚨 Critical Error: Could not import LangGraph workflow from `Graph/edges.py`.")
    with st.expander("Inspect Full Import Traceback"):
        st.code(workflow_import_error)
    st.stop()

# -----------------------------------------------------------------------------
# 10. Research Benchmark Preset Buttons
# -----------------------------------------------------------------------------
st.markdown("##### 🧪 Conference Demo Scenarios (One-Click Test Cases)")
col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    if st.button("🟢 Case 1: Correct Retrieval", use_container_width=True, help="Query directly addressed in internal document"):
        st.session_state.preset_query = "What is backpropagation in neural networks and how does it compute gradients?"

with col_p2:
    if st.button("🟡 Case 2: Ambiguous Retrieval", use_container_width=True, help="Query partially covered; triggers hybrid document + web augmentation"):
        st.session_state.preset_query = "How does deep learning relate to modern DevOps and automated MLOps pipelines?"

with col_p3:
    if st.button("🔴 Case 3: Incorrect Retrieval", use_container_width=True, help="Query absent in document; triggers full web search correction"):
        st.session_state.preset_query = "why is devOps called a low code field and what is GitOps?"

# -----------------------------------------------------------------------------
# 11. Pipeline Visual Architecture Expander
# -----------------------------------------------------------------------------
with st.expander("🗺️ Interactive CRAG Graph Flowchart & Node Topology", expanded=False):
    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        st.graphviz_chart(generate_crag_graphviz(), use_container_width=True)
    with col_g2:
        st.markdown(
            """
            #### 🧭 Architectural Routing Logic:
            
            1. **Multi-Query Retrieval**: Expands initial user query into multiple variations to capture diverse semantic document chunks.
            2. **Retrieval Evaluator**: Evaluates retrieved chunks against scoring criteria ($0.0 - 1.0$).
               - **Correct ($\ge 0.7$)**: Triggers document knowledge strip refinement $\to$ synthesizes answer strictly from internal docs.
               - **Ambiguous ($0.3 - 0.7$)**: Triggers Tavily web search agent $\to$ refines both document and web strips $\to$ merges in hybrid synthesis.
               - **Incorrect ($< 0.3$)**: Documents deemed irrelevant $\to$ triggers Tavily search $\to$ synthesizes answer purely from verified web knowledge.
            3. **Knowledge Refinement**: Decomposes chunks into atomic sentences and filters out irrelevant noise, ensuring factual grounding.
            """
        )

# -----------------------------------------------------------------------------
# 12. Query Input & Execution Engine
# -----------------------------------------------------------------------------
user_query = st.chat_input("Ask a question to trigger the Corrective RAG pipeline...")

# Handle preset queries
if st.session_state.preset_query and not user_query:
    user_query = st.session_state.preset_query
    st.session_state.preset_query = ""

if user_query:
    # Append user question to chat
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Render User Message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(f"**{user_query}**")

    # Render Assistant Execution with Live Animated LangGraph State Tracker
    with st.chat_message("assistant", avatar="🧬"):
        pipeline_status = st.status("🚀 Initializing Corrective RAG Graph...", expanded=True)
        
        # Split live layout: Live Graphviz on left, Live Log on right
        viz_col, log_col = pipeline_status.columns([1.2, 1.8])
        graph_slot = viz_col.empty()
        log_slot = log_col.empty()

        graph_slot.graphviz_chart(generate_crag_graphviz(), use_container_width=True)

        visited_nodes: List[str] = []
        log_records: List[str] = []
        final_state_snapshot: Dict[str, Any] = {}
        error_encountered = None
        start_time = time.time()

        try:
            for node_name, node_output, state_snapshot in execute_crag_pipeline(
                user_query, st.session_state.doc_path
            ):
                visited_nodes.append(node_name)
                final_state_snapshot = state_snapshot
                curr_status = state_snapshot.get("status", "unknown")

                # Node info lookup
                node_info = NODE_METADATA.get(node_name, {"icon": "⚙️", "title": node_name, "desc": ""})
                summary = format_node_output_summary(node_name, node_output)
                
                log_records.append(
                    f"{node_info['icon']} **{node_info['title']}**\n\n"
                    f"<span style='font-size:0.83rem; color:#94A3B8;'>{summary}</span>"
                )

                # Update live graph visualization & execution log
                graph_slot.graphviz_chart(
                    generate_crag_graphviz(
                        current_node=node_name,
                        visited_nodes=visited_nodes,
                        status_state=curr_status,
                    ),
                    use_container_width=True,
                )
                log_slot.markdown("\n\n---\n\n".join(log_records), unsafe_allow_html=True)
                pipeline_status.update(label=f"Executing: {node_info['title']} ({len(visited_nodes)} steps)...")

        except Exception as e:
            error_encountered = traceback.format_exc()

        elapsed_time = time.time() - start_time

        if error_encountered:
            pipeline_status.update(
                label=f"❌ Pipeline halted with error after {elapsed_time:.2f}s",
                state="error",
                expanded=True,
            )
            st.error("Execution encountered an error during LangGraph execution.")
            with st.expander("Full Error Diagnostics"):
                st.code(error_encountered)
        else:
            status_val = final_state_snapshot.get("status", "completed").upper()
            pipeline_status.update(
                label=f"✅ Pipeline Completed in {elapsed_time:.2f}s · Route: {status_val} · {len(visited_nodes)} Nodes Executed",
                state="complete",
                expanded=False,
            )

        # Store latest run results
        st.session_state.latest_run = {
            "query": user_query,
            "state": final_state_snapshot,
            "visited_nodes": visited_nodes,
            "log_records": log_records,
            "elapsed_time": elapsed_time,
            "error": error_encountered,
        }

        # Append assistant response to history
        resp_text = final_state_snapshot.get("response", "No response generated.")
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": resp_text,
                "run_data": st.session_state.latest_run,
            }
        )

# -----------------------------------------------------------------------------
# 13. Comprehensive Results & Deep-Dive Inspection Dashboard
# -----------------------------------------------------------------------------
if st.session_state.latest_run:
    run = st.session_state.latest_run
    state_data = run["state"]
    status_label = state_data.get("status", "unknown").lower()
    final_response = state_data.get("response", "No answer generated.")
    scores = state_data.get("scores", [])
    good_docs = state_data.get("good_docs", [])
    retrieved_docs = state_data.get("retrieved_docs", [])
    refined_doc_context = state_data.get("refined_docs_context", "")
    refined_web_context = state_data.get("refined_web_context", "")
    messages = state_data.get("messages", [])

    st.markdown("### 🎯 Grounded Response & CRAG Decision Analytics")

    # Metrics Summary Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        if status_label == "correct":
            st.markdown(
                """
                <div class="metric-card" style="border-left:4px solid #22C55E;">
                    <div class="metric-label">Routing Decision</div>
                    <div class="metric-value" style="color:#4ADE80;">🟢 CORRECT</div>
                    <span style="font-size:0.75rem; color:#94A3B8;">Doc Confidence &ge; 0.70</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif status_label == "ambigious":
            st.markdown(
                """
                <div class="metric-card" style="border-left:4px solid #F59E0B;">
                    <div class="metric-label">Routing Decision</div>
                    <div class="metric-value" style="color:#FBBF24;">🟡 AMBIGUOUS</div>
                    <span style="font-size:0.75rem; color:#94A3B8;">0.30 &le; Score &lt; 0.70</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="metric-card" style="border-left:4px solid #EF4444;">
                    <div class="metric-label">Routing Decision</div>
                    <div class="metric-value" style="color:#F87171;">🔴 INCORRECT</div>
                    <span style="font-size:0.75rem; color:#94A3B8;">Doc Confidence &lt; 0.30</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_m2:
        max_score = f"{max(scores):.2f}" if scores else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Peak Chunk Relevance</div>
                <div class="metric-value">{max_score}</div>
                <span style="font-size:0.75rem; color:#94A3B8;">Upper Threshold: 0.70</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        source_badge = "📄 Internal Docs" if status_label == "correct" else ("🔀 Hybrid Docs + Web" if status_label == "ambigious" else "🌐 Web Search Only")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Knowledge Synthesis Source</div>
                <div class="metric-value" style="font-size:1.1rem; padding-top:4px;">{source_badge}</div>
                <span style="font-size:0.75rem; color:#94A3B8;">Verified Grounding</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Execution Latency</div>
                <div class="metric-value">{run['elapsed_time']:.2f}s</div>
                <span style="font-size:0.75rem; color:#94A3B8;">{len(run['visited_nodes'])} Graph Nodes</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Synthesized Answer Card
    st.markdown(
        f"""
        <div class="response-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#38BDF8;">
                    ⚡ Synthesized Grounded Output
                </span>
                <span style="font-size:0.75rem; color:#94A3B8;">Query: "{run['query']}"</span>
            </div>
            <div style="font-size:1.02rem; line-height:1.65; color:#F8FAFC;">
                {final_response}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 14. Deep-Dive Inspection Tabs
    # -------------------------------------------------------------------------
    tab_eval, tab_refine, tab_web, tab_trace, tab_docs = st.tabs(
        [
            "⚖️ Evaluator & Confidence Scoring",
            "✂️ Knowledge Refinement (Strips)",
            "🌐 Web Augmentation (Tavily)",
            "🔬 LangGraph State Trace",
            "📄 Raw Retrieved Chunks",
        ]
    )

    # --- TAB 1: EVALUATOR & CONFIDENCE SCORING ---
    with tab_eval:
        st.markdown("#### ⚖️ Document Retrieval Evaluation Matrix")
        st.caption(
            "The CRAG Evaluator scores each candidate chunk against the query ($0.0 - 1.0$). "
            "Chunks $\\ge 0.7$ trigger internal generation; $0.3 - 0.7$ trigger hybrid augmentation; $< 0.3$ trigger web search fallback."
        )

        if scores and retrieved_docs:
            for idx, (doc, score) in enumerate(zip(retrieved_docs, scores)):
                if score >= 0.7:
                    badge_class = "score-high"
                    badge_label = f"HIGH RELEVANCE ({score:.2f}) - KEPT"
                elif score >= 0.3:
                    badge_class = "score-mid"
                    badge_label = f"AMBIGUOUS / WEAK ({score:.2f}) - FILTERED"
                else:
                    badge_class = "score-low"
                    badge_label = f"IRRELEVANT ({score:.2f}) - DISCARDED"

                with st.expander(f"Chunk #{idx+1} · Score: {score:.2f} — {badge_label}", expanded=(idx == 0)):
                    st.markdown(
                        f"<span class='score-badge {badge_class}'>Score: {score:.2f}</span> "
                        f"<span style='font-size:0.8rem; color:#94A3B8; margin-left:10px;'>Length: {len(doc.page_content)} characters</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"```text\n{doc.page_content.strip()}\n```")
        else:
            st.info("No chunk evaluations recorded for this execution.")

    # --- TAB 2: KNOWLEDGE REFINEMENT (STRIPS) ---
    with tab_refine:
        st.markdown("#### ✂️ Sentence-Level Knowledge Refinement")
        st.caption(
            "To eliminate noise and hallucination risks, CRAG decomposes relevant document chunks "
            "into atomic sentences and applies strict relevance filtering to retain only essential knowledge strips."
        )

        if refined_doc_context:
            st.markdown("##### 🟢 Kept Internal Knowledge Strips:")
            sentences = [s.strip() for s in refined_doc_context.split(". ") if s.strip()]
            for s in sentences:
                st.markdown(f"<div class='strip-item'>{s}.</div>", unsafe_allow_html=True)
        else:
            st.info("No internal document strips were retained (routed to web search fallback).")

    # --- TAB 3: WEB AUGMENTATION (TAVILY) ---
    with tab_web:
        st.markdown("#### 🌐 Tavily Web Search Augmentation")
        st.caption(
            "When internal document retrieval is ambiguous or incorrect, CRAG autonomously dispatches a web search agent."
        )

        if status_label in ("ambigious", "incorrect"):
            if refined_web_context:
                st.markdown("##### 🟣 Refined Web Knowledge Strips:")
                web_sentences = [s.strip() for s in refined_web_context.split(". ") if s.strip()]
                for s in web_sentences:
                    st.markdown(f"<div class='strip-item strip-web'>{s}.</div>", unsafe_allow_html=True)

            if messages:
                with st.expander("📡 Raw Tavily Tool Messages & Search Results"):
                    for msg in messages:
                        if hasattr(msg, "content"):
                            st.write(msg.content)
                        else:
                            st.write(msg)
        else:
            st.success("Internal document confidence was sufficient ($\\ge 0.7$). Web search augmentation was not required!")

    # --- TAB 4: LANGGRAPH STATE TRACE ---
    with tab_trace:
        st.markdown("#### 🔬 Full LangGraph State Snapshot & Execution Log")
        
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("##### 📋 Visited Node Sequence:")
            st.code(" ➔ ".join(run["visited_nodes"]))
            
            st.markdown("##### ⏱️ Execution Timeline:")
            for record in run["log_records"]:
                st.markdown(record, unsafe_allow_html=True)

        with col_t2:
            st.markdown("##### 📦 State Dictionary:")
            cleaned_state = {}
            for k, v in state_data.items():
                if k in ("database", "chunks"):
                    cleaned_state[k] = f"<{type(v).__name__} Object>"
                elif k in ("good_docs", "retrieved_docs"):
                    cleaned_state[k] = [f"<Document len={len(d.page_content)}>" for d in v] if isinstance(v, list) else str(v)
                elif k == "messages":
                    cleaned_state[k] = [str(m) for m in v] if isinstance(v, list) else str(v)
                else:
                    cleaned_state[k] = v
            st.json(cleaned_state)

    # --- TAB 5: RAW RETRIEVED CHUNKS ---
    with tab_docs:
        st.markdown("#### 📄 Raw Multi-Query Retrieved Chunks")
        st.caption("Candidate document chunks retrieved from ChromaDB prior to confidence evaluation.")

        if retrieved_docs:
            for idx, doc in enumerate(retrieved_docs):
                with st.expander(f"Raw Chunk #{idx+1}"):
                    st.markdown(f"```text\n{doc.page_content}\n```")
                    if hasattr(doc, "metadata") and doc.metadata:
                        st.json(doc.metadata)
        else:
            st.info("No raw chunks retrieved.")

# -----------------------------------------------------------------------------
# 15. Previous Chat History
# -----------------------------------------------------------------------------
if len(st.session_state.chat_history) > 2:
    st.markdown("---")
    with st.expander(f"📜 Conversation History ({len(st.session_state.chat_history)} messages)", expanded=False):
        for msg in st.session_state.chat_history[:-2]:
            role = msg["role"]
            avatar = "🧑‍💻" if role == "user" else "🧬"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])