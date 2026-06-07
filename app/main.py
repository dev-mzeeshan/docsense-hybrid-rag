"""
main.py — DocSense: Hybrid RAG Document Intelligence
======================================================
What sets this apart from a standard RAG Streamlit demo:

  ✦ Multi-document  — upload multiple PDFs, query across all of them.
  ✦ Hybrid Search   — toggle between BM25+FAISS ensemble vs. pure semantic.
  ✦ Memory          — follow-up questions use actual conversation history,
                      not just a display log.
  ✦ Analytics panel — chunk count, pages, word estimate per document.
  ✦ Source viewer   — expander shows page number + preview of each retrieved chunk.

Run:  streamlit run main.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from rag_pipeline import (
    EMBED_MODEL,
    create_qa_chain,
    create_vector_store,
    get_document_stats,
    load_and_split_pdf,
    merge_into_store,
)

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocSense - Hybrid RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Base */
.main { background-color: #0d1117; }
section[data-testid="stSidebar"] { background-color: #161b22; }

/* Buttons */
.stButton > button {
    border-radius: 8px; height: 2.6em;
    background: #1f6feb; color: #fff; border: none;
    font-weight: 600; transition: background 0.2s;
}
.stButton > button:hover { background: #388bfd; }

/* Metrics */
[data-testid="stMetricValue"] { font-size: 1.5rem; color: #388bfd; }

/* Source cards */
.src-card {
    background: #161b22;
    border-left: 3px solid #1f6feb;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.83rem;
    color: #8b949e;
    line-height: 1.5;
}
.src-label { color: #58a6ff; font-weight: 600; font-size: 0.78rem; }

/* Mode badge */
.mode-badge {
    display: inline-block;
    background: #1f6feb22;
    border: 1px solid #1f6feb66;
    color: #58a6ff;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "messages":     [],           # display messages [{role, content, sources?}]
    "chat_history": [],           # LangChain Message objects for history-aware retrieval
    "qa_chain":     None,         # active RAG chain
    "vector_store": None,         # merged FAISS index
    "all_chunks":   [],           # all chunks across every uploaded PDF
    "doc_stats":    [],           # per-doc stats dicts
    "doc_names":    [],           # uploaded file names (dedup guard)
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    selected_model = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        help="70b → more accurate | 8b → faster responses",
    )

    use_hybrid = st.toggle(
        "Hybrid Search  (BM25 + FAISS)",
        value=True,
        help=(
            "**ON** - combines keyword (BM25) and semantic (FAISS) retrieval.\n"
            "Recommended: catches exact terms that vector search misses.\n\n"
            "**OFF** - pure semantic / vector search only."
        ),
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2,
                            help="Lower = more factual. Higher = more creative.")
    chunk_size  = st.slider("Chunk Size (chars)", 500, 2000, 1000)

    st.divider()

    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear Chat"):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.rerun()

    if col2.button("♻️ Reset All"):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    # ── Per-document analytics ────────────────────────────────────────────────
    if st.session_state.doc_stats:
        st.divider()
        st.markdown("#### 📊 Loaded Documents")

        total_chunks = sum(s["chunks"] for s in st.session_state.doc_stats)
        total_pages  = sum(s["pages"]  for s in st.session_state.doc_stats)
        total_words  = sum(s["words"]  for s in st.session_state.doc_stats)

        m1, m2, m3 = st.columns(3)
        m1.metric("Chunks", total_chunks)
        m2.metric("Pages",  total_pages)
        m3.metric("~Words", f"{total_words:,}")

        st.markdown("**Files:**")
        for i, name in enumerate(st.session_state.doc_names):
            s = st.session_state.doc_stats[i]
            st.caption(f"• {name}  ·  {s['chunks']} chunks  ·  {s['pages']} pages")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔍 DocSense - Hybrid RAG")
st.markdown(
    "Upload one or more PDFs. Ask anything. "
    "**Hybrid retrieval** (BM25 + FAISS) delivers higher accuracy than standard RAG "
    "by combining keyword and semantic search."
)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="collapsed",
)

new_files = [f for f in (uploaded_files or []) if f.name not in st.session_state.doc_names]

if new_files:
    for uploaded_file in new_files:
        with st.status(f"⚙️ Processing '{uploaded_file.name}'…", expanded=True) as status:

            st.write("📄 Extracting text…")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                st.write("✂️ Splitting into semantic chunks…")
                chunks = load_and_split_pdf(tmp_path, chunk_size=chunk_size)

                st.write("🔢 Building / updating vector index…")
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = create_vector_store(chunks)
                else:
                    st.session_state.vector_store = merge_into_store(
                        st.session_state.vector_store, chunks
                    )

                st.session_state.all_chunks.extend(chunks)
                st.session_state.doc_stats.append(get_document_stats(chunks))
                st.session_state.doc_names.append(uploaded_file.name)

                st.write("🔗 Configuring retrieval chain…")
                st.session_state.qa_chain = create_qa_chain(
                    vector_store=st.session_state.vector_store,
                    model_name=selected_model,
                    temperature=temperature,
                    chunks=st.session_state.all_chunks,
                    use_hybrid=use_hybrid,
                )

                status.update(
                    label=f"✅ '{uploaded_file.name}' ready!",
                    state="complete",
                    expanded=False,
                )
                st.toast(f"{uploaded_file.name} processed!", icon="🔥")

            except ValueError as e:
                status.update(label="❌ Could not process document", state="error")
                st.error(f"⚠️ {e}")
            except Exception as e:
                status.update(label="❌ Unexpected error", state="error")
                st.error(f"Something went wrong: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Rebuild chain if settings changed (hybrid toggle / model / temperature)
if st.session_state.vector_store is not None and st.session_state.all_chunks:
    st.session_state.qa_chain = create_qa_chain(
        vector_store=st.session_state.vector_store,
        model_name=selected_model,
        temperature=temperature,
        chunks=st.session_state.all_chunks,
        use_hybrid=use_hybrid,
    )

# ── Status bar ────────────────────────────────────────────────────────────────
if st.session_state.qa_chain:
    mode  = "🔀 Hybrid (BM25 + FAISS)" if use_hybrid else "🔷 Semantic Only (FAISS)"
    ndocs = len(st.session_state.doc_names)
    st.markdown(
        f'<span class="mode-badge">{mode}</span>&nbsp;&nbsp;'
        f'<span class="mode-badge">📄 {ndocs} document{"s" if ndocs != 1 else ""} loaded</span>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Chat display ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source chunk(s) used"):
                for i, src in enumerate(msg["sources"]):
                    page    = src.metadata.get("page", "?")
                    source  = src.metadata.get("source", "")
                    preview = src.page_content[:300].replace("\n", " ").strip()
                    st.markdown(
                        f'<div class="src-card">'
                        f'<span class="src-label">Chunk {i+1} - Page {page}'
                        f'{f"  ·  {os.path.basename(source)}" if source else ""}</span><br>'
                        f"{preview}…"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about your documents…"):

    if not st.session_state.qa_chain:
        st.warning("⬆️ Please upload at least one PDF before asking questions.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            response = st.session_state.qa_chain.invoke({
                "input":        prompt,
                "chat_history": st.session_state.chat_history,
            })
            answer  = response["answer"]
            sources = response.get("context", [])

        st.markdown(answer)

        if sources:
            with st.expander(f"📚 {len(sources)} source chunk(s) used"):
                for i, src in enumerate(sources):
                    page    = src.metadata.get("page", "?")
                    source  = src.metadata.get("source", "")
                    preview = src.page_content[:300].replace("\n", " ").strip()
                    st.markdown(
                        f'<div class="src-card">'
                        f'<span class="src-label">Chunk {i+1} - Page {page}'
                        f'{f"  ·  {os.path.basename(source)}" if source else ""}</span><br>'
                        f"{preview}…"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # Update LangChain conversation history (used by history-aware retriever)
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=answer))
    # Keep last 20 messages (~10 turns) to avoid context overflow
    st.session_state.chat_history = st.session_state.chat_history[-20:]

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })