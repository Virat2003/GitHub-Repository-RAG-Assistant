"""
GitHub Repository RAG Assistant
================================
A Streamlit application that lets a user paste a public GitHub repository
URL, indexes the codebase (clone -> load -> chunk -> embed -> vector store),
and then answers natural-language questions about the repository using a
Retrieval-Augmented Generation (RAG) pipeline powered by a Mistral LLM.

UI is styled to feel like a modern AI chat product (ChatGPT / Claude /
Cursor): centered ~800px chat column, left/right message bubbles with
avatars, a welcome screen, and collapsible source citations.

Run with:
    streamlit run app.py
"""

import os

# --------------------------------------------------------------------------
# IMPORTANT: these env vars must be set BEFORE numpy / torch / sentence-
# transformers are imported anywhere (including transitively, via the
# backend modules below). Without this, OpenBLAS spawns one thread per CPU
# core and each thread tries to allocate its own scratch buffer, which can
# exhaust available memory and crash with:
#   "OpenBLAS error: Memory allocation still failed after 10 retries"
# This is especially common on memory-constrained hosts (e.g. Streamlit
# Community Cloud's free tier, small Docker containers, etc.).
# --------------------------------------------------------------------------
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import html
import re
import traceback

import streamlit as st
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Backend pipeline imports (existing project modules — unchanged)
# --------------------------------------------------------------------------
from loader.github_loader import clone_repository
from loader.code_loader import load_repository
from chunking.splitter import split_document
from embeddings.embedding_model import get_embedding_model
from vector_store.chroma_db import create_vector_store
from retriever.retriever import get_retriever
from chains.rag_chain import create_rag_chain

# `get_llm()` is the documented entry point. Some versions of llm/llm.py only
# expose a pre-built `llm` instance instead of a factory function, so we
# support both without changing the backend module.
try:
    from llm.llm import get_llm
except ImportError:
    from llm.llm import llm as _default_llm_instance

    def get_llm():
        return _default_llm_instance


load_dotenv()


# ==========================================================================
# Page configuration
# ==========================================================================
st.set_page_config(
    page_title="GitHub Repository RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# Global CSS — ChatGPT / Claude style chat UI
# ==========================================================================
st.markdown(
    """
    <style>
        /* ---------- Clean up Streamlit chrome ---------- */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        div[data-testid="stDecoration"] { display: none; }
        header[data-testid="stHeader"] { background: transparent; }

        /* ---------- Centered, responsive chat column ---------- */
        .main .block-container {
            max-width: 800px;
            margin: 0 auto;
            padding-top: 1.25rem;
            padding-bottom: 7rem; /* leave room above the fixed chat input */
        }

        /* Center + constrain the fixed bottom chat input to match the
           chat column. Different Streamlit versions use slightly
           different test-ids for this container, so we target a few. */
        div[data-testid="stBottomBlockContainer"],
        div[data-testid="stChatInput"] {
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        div[data-testid="stChatInput"] textarea {
            border-radius: 14px;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.18);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        /* ---------- Buttons: soft, rounded, modern ---------- */
        .stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            transition: all 0.15s ease;
        }
        .stButton > button:hover {
            border-color: rgba(99, 102, 241, 0.6);
            color: #6366f1;
        }

        /* ---------- Repo status badge ---------- */
        .repo-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            background-color: rgba(16, 163, 127, 0.12);
            color: #10a37f;
            font-size: 0.82rem;
            font-weight: 600;
            border: 1px solid rgba(16, 163, 127, 0.3);
        }

        /* ---------- Compact app header above the chat ---------- */
        .app-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0 1rem 0;
            font-size: 0.95rem;
            font-weight: 600;
            color: rgba(128, 128, 128, 0.95);
            border-bottom: 1px solid rgba(128, 128, 128, 0.12);
            margin-bottom: 1rem;
        }
        .app-header .repo-name {
            color: var(--text-color, inherit);
            font-weight: 700;
        }

        /* ---------- Welcome / empty state hero ---------- */
        .hero {
            text-align: center;
            padding: 3.5rem 1rem 1.5rem 1rem;
        }
        .hero .hero-icon {
            font-size: 3rem;
            margin-bottom: 0.75rem;
        }
        .hero h1 {
            font-size: 1.6rem;
            margin-bottom: 0.4rem;
        }
        .hero p {
            color: rgba(128, 128, 128, 0.95);
            font-size: 0.95rem;
            max-width: 480px;
            margin: 0 auto;
        }
        .hero-ready {
            text-align: center;
            padding: 1.5rem 1rem 0.5rem 1rem;
        }
        .hero-ready h3 {
            margin-bottom: 0.2rem;
        }
        .hero-ready p {
            color: rgba(128, 128, 128, 0.95);
            font-size: 0.9rem;
        }

        /* ---------- Chat bubbles ---------- */
        .chat-row {
            display: flex;
            gap: 0.6rem;
            align-items: flex-end;
            margin: 0.9rem 0;
            animation: fadeIn 0.2s ease;
        }
        .chat-row.user { flex-direction: row-reverse; }

        .avatar {
            width: 32px;
            height: 32px;
            min-width: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            color: #fff;
        }
        .avatar.user-avatar {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
        }
        .avatar.assistant-avatar {
            background: linear-gradient(135deg, #10a37f, #0d8d6c);
        }

        .bubble {
            max-width: 76%;
            padding: 0.7rem 1rem;
            border-radius: 1.1rem;
            font-size: 0.95rem;
            line-height: 1.6;
            word-wrap: break-word;
        }
        .bubble.user-bubble {
            background: #6366f1;
            color: #ffffff;
            border-bottom-right-radius: 0.3rem;
            white-space: pre-wrap;
        }
        .bubble.assistant-bubble {
            background: var(--secondary-background-color, #f4f4f6);
            color: var(--text-color, #111111);
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-bottom-left-radius: 0.3rem;
        }
        .bubble.assistant-bubble code {
            background: rgba(128, 128, 128, 0.18);
            padding: 0.1rem 0.35rem;
            border-radius: 4px;
            font-size: 0.85em;
        }
        .bubble.assistant-bubble pre {
            background: rgba(128, 128, 128, 0.14);
            padding: 0.75rem;
            border-radius: 8px;
            overflow-x: auto;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ---------- Source citation expander ---------- */
        .main div[data-testid="stExpander"] {
            max-width: 76%;
            margin: -0.5rem 0 0.9rem 2.55rem; /* align under assistant bubble */
            border-radius: 0.7rem;
            font-size: 0.85rem;
        }
        .source-item {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 0.82rem;
            padding: 0.15rem 0;
            color: rgba(128, 128, 128, 0.95);
        }

        /* ---------- Suggestion chips ---------- */
        div[data-testid="column"] .stButton > button {
            font-size: 0.85rem;
            padding: 0.5rem 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Session state initialization
# ==========================================================================
def init_session_state() -> None:
    """Set up default values in st.session_state on first run."""
    defaults = {
        "rag_chain": None,
        "retriever": None,
        "chat_history": [],  # list of {"role", "content", "sources"(optional)}
        "repo_processed": False,
        "current_repo_url": None,
        "repo_path": None,
        "num_docs": 0,
        "num_chunks": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ==========================================================================
# Helper functions
# ==========================================================================
GITHUB_URL_PATTERN = re.compile(
    r"^https?://(www\.)?github\.com/[\w.\-]+/[\w.\-]+(\.git)?/?$"
)


def is_valid_github_url(url: str) -> bool:
    """Basic validation that the input looks like a GitHub repo URL."""
    return bool(GITHUB_URL_PATTERN.match(url.strip()))


def extract_answer(response) -> str:
    """
    Normalize the rag_chain.invoke(...) response into a plain string,
    regardless of whether the chain returns a raw string or a dict-like
    object (common with LangChain chains that return {"answer": ...} etc.)
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("answer", "result", "output_text", "output", "text"):
            if key in response and isinstance(response[key], str):
                return response[key]
        return str(response)
    return str(response)


def get_source_paths(docs) -> list:
    """Extract a de-duplicated, ordered list of file paths from retrieved docs."""
    sources = []
    for doc in docs or []:
        file_path = doc.metadata.get("file_path", "Unknown source")
        if file_path not in sources:
            sources.append(file_path)
    return sources


def reset_repository_state() -> None:
    """Clear out indexing + chat state so a new repository can be processed."""
    st.session_state.rag_chain = None
    st.session_state.retriever = None
    st.session_state.repo_processed = False
    st.session_state.current_repo_url = None
    st.session_state.repo_path = None
    st.session_state.num_docs = 0
    st.session_state.num_chunks = 0
    st.session_state.chat_history = []


def process_repository(repo_url: str) -> bool:
    """
    Run the full ingestion pipeline:
    clone -> load -> chunk -> embed -> vector store -> retriever -> rag_chain

    On success, stores the rag_chain (and repo metadata) in st.session_state.
    Returns True on success, False on failure.
    """
    try:
        with st.status("Processing repository…", expanded=True) as status:

            status.write("🔄 Cloning repository…")
            repo_path = clone_repository(repo_url)
            status.write("✅ Repository cloned.")

            status.write("📂 Loading code files…")
            documents = load_repository(repo_path)
            if not documents:
                status.update(label="No files found", state="error")
                st.error(
                    "No supported code files were found in this repository. "
                    "Please check the URL and try again."
                )
                return False
            status.write(f"✅ Loaded {len(documents)} file(s).")

            status.write("✂️ Splitting documents into chunks…")
            chunks = split_document(documents)
            status.write(f"✅ Created {len(chunks)} chunk(s).")

            status.write("🧠 Loading HuggingFace embedding model…")
            embedding_model = get_embedding_model()
            status.write("✅ Embedding model ready.")

            status.write("🗄️ Generating embeddings and building ChromaDB vector store…")
            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            vector_store = create_vector_store(chunks, embedding_model, repo_name)
            status.write("✅ Vector store created.")

            status.write("🔍 Creating retriever…")
            retriever = get_retriever(vector_store)
            status.write("✅ Retriever ready.")

            status.write("🔗 Connecting retriever to Mistral LLM (RAG chain)…")
            llm = get_llm()
            rag_chain = create_rag_chain(retriever, llm)
            status.write("✅ RAG chain ready.")

            # Persist everything needed for the chat session
            st.session_state.rag_chain = rag_chain
            st.session_state.retriever = retriever
            st.session_state.repo_processed = True
            st.session_state.current_repo_url = repo_url
            st.session_state.repo_path = repo_path
            st.session_state.num_docs = len(documents)
            st.session_state.num_chunks = len(chunks)
            st.session_state.chat_history = []

            status.update(
                label="Repository indexed successfully!", state="complete"
            )

        return True

    except Exception as exc:  # noqa: BLE001 - surface any backend failure to the UI
        st.session_state.rag_chain = None
        st.session_state.repo_processed = False
        st.error(f"❌ Failed to process repository: {exc}")
        with st.expander("Show technical details"):
            st.code(traceback.format_exc())
        return False


def render_bubble(role: str, content: str) -> None:
    """Render a single chat message as a ChatGPT/Claude-style bubble."""
    if role == "user":
        safe_content = html.escape(content)
        st.markdown(
            f"""
            <div class="chat-row user">
                <div class="avatar user-avatar">👤</div>
                <div class="bubble user-bubble">{safe_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Assistant content is left un-escaped so markdown/code formatting
        # from the LLM (bold, inline code, fenced code blocks) still renders.
        st.markdown(
            f"""
            <div class="chat-row assistant">
                <div class="avatar assistant-avatar">🤖</div>
                <div class="bubble assistant-bubble">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sources(sources: list) -> None:
    """Render a collapsible '📂 Sources' expander below an assistant bubble."""
    if not sources:
        return
    with st.expander(f"📂 Sources ({len(sources)})"):
        for path in sources:
            st.markdown(f'<div class="source-item">📄 {html.escape(path)}</div>', unsafe_allow_html=True)


# ==========================================================================
# Sidebar — Repository input & processing controls
# ==========================================================================
with st.sidebar:
    st.markdown("## 📦 Repository Setup")
    st.caption("Paste a public GitHub repository URL to index its codebase.")

    repo_url_input = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo.git",
        key="repo_url_input",
    )

    process_clicked = st.button(
        "🚀 Process Repository",
        use_container_width=True,
        type="primary",
    )

    if process_clicked:
        url = repo_url_input.strip()
        if not url:
            st.error("⚠️ Please enter a GitHub repository URL.")
        elif not is_valid_github_url(url):
            st.error(
                "⚠️ That doesn't look like a valid GitHub repository URL.\n\n"
                "Expected format: `https://github.com/owner/repo`"
            )
        else:
            success = process_repository(url)
            if success:
                st.rerun()

    st.divider()

    # ---- Status panel -----------------------------------------------------
    if st.session_state.repo_processed:
        st.markdown(
            '<span class="repo-badge">✅ Repository Indexed</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Repo:** {st.session_state.current_repo_url}")

        col1, col2 = st.columns(2)
        col1.metric("Files", st.session_state.num_docs)
        col2.metric("Chunks", st.session_state.num_chunks)

        st.write("")
        if st.button("🔄 Process a Different Repository", use_container_width=True):
            reset_repository_state()
            st.rerun()

        if st.session_state.chat_history and st.button(
            "🆕 New Chat", use_container_width=True
        ):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No repository indexed yet. Enter a URL above to get started.")

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown(
            "1. **Clone** the repository\n"
            "2. **Load** all code files recursively\n"
            "3. **Chunk** the source code\n"
            "4. **Embed** chunks with a HuggingFace model\n"
            "5. **Store** vectors in ChromaDB\n"
            "6. **Retrieve + Answer** using a Mistral LLM"
        )


# ==========================================================================
# Main Area — Chat interface
# ==========================================================================
repo_processed = st.session_state.repo_processed

if repo_processed:
    repo_display_name = (
        st.session_state.current_repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    )
    st.markdown(
        f"""
        <div class="app-header">
            🤖 GitHub Repository RAG Assistant
            &nbsp;·&nbsp; 📦 <span class="repo-name">{html.escape(repo_display_name)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="app-header">🤖 GitHub Repository RAG Assistant</div>',
        unsafe_allow_html=True,
    )

if not repo_processed:
    # ---- Welcome / onboarding screen ---------------------------------------
    st.markdown(
        """
        <div class="hero">
            <div class="hero-icon">🤖</div>
            <h1>Chat with any GitHub repository</h1>
            <p>Paste a public GitHub repository URL in the sidebar and click
            <b>Process Repository</b>. Once it's indexed, ask anything about
            the codebase — architecture, specific functions, how features
            work, and more.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # ---- Render existing chat history --------------------------------------
    for message in st.session_state.chat_history:
        render_bubble(message["role"], message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

    # ---- Empty-chat state: ready prompt + starter suggestions --------------
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="hero-ready">
                <h3>✅ Repository ready</h3>
                <p>Ask me anything about this codebase, or try a suggestion below.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        suggestions = [
            "What does this project do?",
            "Explain the overall architecture.",
            "How does authentication work here?",
            "Which frameworks or libraries are used?",
        ]
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            if col.button(suggestion, use_container_width=True):
                st.session_state["_pending_question"] = suggestion

# ==========================================================================
# Chat input (disabled until a repository has been processed)
# ==========================================================================
pending_question = st.session_state.pop("_pending_question", None)

user_question = st.chat_input(
    "Ask a question about the repository…"
    if repo_processed
    else "Process a repository first to start chatting…",
    disabled=not repo_processed,
)

question = user_question or pending_question

if question and repo_processed:
    # Show the user's question immediately
    render_bubble("user", question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Generate and show the assistant's answer
    with st.spinner("Searching the codebase and generating an answer…"):
        sources = []
        try:
            docs = st.session_state.retriever.invoke(question)
            sources = get_source_paths(docs)

            raw_response = st.session_state.rag_chain.invoke(question)
            answer = extract_answer(raw_response)
        except Exception as exc:  # noqa: BLE001
            answer = f"⚠️ Sorry, I couldn't generate an answer: {exc}"

    render_bubble("assistant", answer)
    render_sources(sources)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )