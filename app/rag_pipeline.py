"""
rag_pipeline.py — Hybrid RAG Core
===================================
What makes this different from a standard RAG tutorial:

1. Hybrid Retrieval  — BM25 (keyword) + FAISS (semantic) ensemble.
   Pure vector search fails on exact terms (names, codes, numbers).
   Pure BM25 fails on paraphrased or conceptual queries.
   Hybrid covers both failure modes.

2. MMR (Maximum Marginal Relevance) — retrieved chunks are diverse,
   not just multiple copies of the same passage.

3. History-Aware Retriever — before searching, the chain rephrases
   follow-up questions into standalone queries using conversation history.
   This is the difference between a chatbot and a toy.

4. Multi-document — all docs share one FAISS index; new docs are merged in.

Dependencies:
    pip install rank_bm25               # BM25Retriever
    pip install langchain langchain-community langchain-groq langchain-huggingface
    pip install faiss-cpu sentence-transformers
"""

import os
from typing import Optional

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── Shared embedding model ────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings  = None   # lazy singleton — avoids reloading weights on every call


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


# ── Document loading ──────────────────────────────────────────────────────────

def load_and_split_pdf(pdf_path: str,
                       chunk_size: int = 1000,
                       chunk_overlap: int = 200) -> list:
    """
    Load a PDF and split into overlapping text chunks.

    Separators are tried in order (paragraph → sentence → word) so the
    splitter preserves semantic boundaries wherever possible.
    """
    loader = PyPDFium2Loader(pdf_path)
    docs    = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
    )
    return splitter.split_documents(docs)


def _validate_chunks(chunks: list) -> list:
    """Drop empty / noise chunks that would corrupt the FAISS index."""
    valid = [c for c in chunks if c.page_content and len(c.page_content.strip()) > 20]
    if not valid:
        raise ValueError(
            "No readable text found in this PDF. "
            "The file may be image-based or scanned. "
            "Please upload a text-based PDF."
        )
    dropped = len(chunks) - len(valid)
    if dropped:
        print(f"[RAG] Dropped {dropped} empty chunks ({len(valid)} kept).")
    return valid


# ── Vector store ──────────────────────────────────────────────────────────────

def create_vector_store(chunks: list) -> FAISS:
    """Embed validated chunks and build a FAISS index."""
    valid = _validate_chunks(chunks)
    return FAISS.from_documents(valid, _get_embeddings())


def merge_into_store(existing_store: FAISS, new_chunks: list) -> FAISS:
    """
    Merge new document chunks into an existing FAISS index.
    Avoids rebuilding embeddings for all prior documents.
    """
    valid = _validate_chunks(new_chunks)
    new_store = FAISS.from_documents(valid, _get_embeddings())
    existing_store.merge_from(new_store)
    return existing_store


# ── Retrievers ────────────────────────────────────────────────────────────────

def create_hybrid_retriever(chunks: list,
                            vector_store: FAISS,
                            k: int = 5) -> EnsembleRetriever:
    """
    Ensemble retriever: 35% BM25 keyword + 65% FAISS semantic (MMR).

    Why 35 / 65?
    - FAISS handles paraphrased queries better (higher weight).
    - BM25 rescues queries containing specific codes, names, or rare terms.
    - MMR (lambda_mult=0.7) balances relevance vs. diversity in FAISS results,
      preventing three chunks from the same sentence dominating the context.

    Requires:  pip install rank_bm25
    """
    valid = _validate_chunks(chunks)

    bm25 = BM25Retriever.from_documents(valid)
    bm25.k = k

    faiss = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 2, "lambda_mult": 0.7},
    )

    return EnsembleRetriever(
        retrievers=[bm25, faiss],
        weights=[0.35, 0.65],
    )


# ── QA chain ──────────────────────────────────────────────────────────────────

def create_qa_chain(vector_store: FAISS,
                    model_name: str = "llama-3.3-70b-versatile",
                    temperature: float = 0.2,
                    chunks: Optional[list] = None,
                    use_hybrid: bool = True):
    """
    Build a conversational, history-aware RAG chain.

    Pipeline:
        user question + chat_history
              ↓
        [History-Aware Retriever]   ← rephrases follow-ups as standalone queries
              ↓
        [Retriever: Hybrid | FAISS] ← fetches top-k relevant chunks
              ↓
        [LLM: Groq]                 ← generates grounded answer
              ↓
        answer + source chunks

    Args:
        vector_store:  FAISS index of all uploaded documents.
        model_name:    Groq model identifier.
        temperature:   Generation temperature (0 = deterministic/factual).
        chunks:        All document chunks — required for BM25 in hybrid mode.
        use_hybrid:    True → BM25 + FAISS ensemble. False → pure FAISS (MMR).

    Returns:
        LangChain retrieval chain.
        Call with: chain.invoke({"input": query, "chat_history": [...]})
    """
    llm = ChatGroq(model_name=model_name, temperature=temperature)

    # ── Retriever ─────────────────────────────────────────────────────────────
    if use_hybrid and chunks:
        base_retriever = create_hybrid_retriever(chunks, vector_store)
    else:
        base_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7},
        )

    # ── Step 1: Rephrase follow-up questions ──────────────────────────────────
    # Without this, "What about the second point?" would retrieve nothing useful.
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a query reformulation assistant. "
         "Given the conversation history and the user's latest message, "
         "produce a standalone question that captures the user's intent "
         "without requiring the conversation history to understand it. "
         "If the message is already standalone, return it unchanged. "
         "Output only the reformulated question — no explanation, no prefix."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, base_retriever, contextualize_prompt
    )

    # ── Step 2: Answer grounded in retrieved context ──────────────────────────
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise document analysis assistant. "
         "Answer ONLY using the context provided below. "
         "If the answer cannot be found in the context, respond with: "
         "'This information is not found in the uploaded document(s).' "
         "When possible, mention the relevant section or page number. "
         "Be concise and accurate.\n\n"
         "── Relevant Context ──\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    combine_chain = create_stuff_documents_chain(llm, answer_prompt)
    return create_retrieval_chain(history_aware_retriever, combine_chain)


# ── Document analytics ────────────────────────────────────────────────────────

def get_document_stats(chunks: list) -> dict:
    """
    Return a quick analytics summary for a processed document.
    Used to populate the sidebar dashboard in main.py.
    """
    valid = [c for c in chunks if c.page_content and len(c.page_content.strip()) > 20]
    if not valid:
        return {"chunks": 0, "pages": 0, "words": 0, "avg_chunk_chars": 0}

    total_chars = sum(len(c.page_content) for c in valid)
    pages       = {c.metadata.get("page", 0) for c in valid}

    return {
        "chunks":          len(valid),
        "pages":           len(pages),
        "words":           total_chars // 5,         # rough estimate
        "avg_chunk_chars": total_chars // len(valid),
    }