<div align="center">

# 🔍 DocSense

### Hybrid RAG Document Intelligence

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3-F55036?style=flat)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0064DB?style=flat)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)](LICENSE)

**Upload multiple PDFs. Ask complex questions. Get cited, accurate answers.**

Most RAG demos use only semantic search. DocSense combines **BM25 keyword search + FAISS vector search** in a weighted ensemble — covering failure cases that either approach alone misses.

[Live Demo →](https://docsense-hybrid-rag.streamlit.app) · [Report Bug](https://github.com/dev-mzeeshan/docsense-hybrid-rag/issues) · [LinkedIn](https://linkedin.com/in/zeeshanofficial)

</div>

---

## The Problem with Standard RAG

Every RAG tutorial gives you the same thing: embed your PDF → store in FAISS → retrieve top-k by cosine similarity → feed to LLM.

This breaks in two common situations:

| Failure Case | Example Query | Why It Fails |
|---|---|---|
| **Exact terms** | *"What is clause 4.2.1?"* | Vector search finds semantically similar text, not the exact clause label |
| **Rare vocabulary** | *"Show me the EBITDA figure"* | The embedding model may not align "EBITDA" precisely with its context |
| **Follow-up questions** | *"What about the second point?"* | Without history awareness, the retriever has no context to search |

DocSense addresses all three.

---

## Solution: Hybrid Retrieval

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│   History-Aware Rephrasing      │  ← Converts "what about the second point?"
│   (LLM + Chat History)          │    into a standalone, searchable question
└─────────────┬───────────────────┘
              │
    ┌─────────▼─────────┐
    │                   │
    ▼                   ▼
┌──────────┐     ┌─────────────────┐
│  BM25    │     │  FAISS (MMR)    │  MMR = Maximum Marginal Relevance
│ Keyword  │     │  Semantic       │  ensures diverse, non-redundant chunks
│  35%     │     │    65%          │
└────┬─────┘     └────────┬────────┘
     │                    │
     └─────────┬──────────┘
               │  Reciprocal Rank Fusion
               ▼
      Top-K Merged Chunks
               │
               ▼
    ┌──────────────────┐
    │  Groq LLM        │  ← Answers ONLY from retrieved context
    │  (Llama 3.3 70B) │    Cites page numbers
    └──────────────────┘
               │
               ▼
    Answer + Source Citations
```

**Why 35 / 65 weights?**

BM25 handles exact term matching but fails on paraphrasing. FAISS handles semantics but drifts on specific codes, numbers, or names. The 35/65 split gives FAISS the higher weight for conceptual queries while letting BM25 rescue precision-critical lookups. Weights are configurable in the sidebar.

---

## Features

- **Hybrid Retrieval** — BM25 + FAISS ensemble, togglable from the UI
- **Multi-document** — Upload multiple PDFs; all share one merged FAISS index
- **Conversational Memory** — History-aware retriever rephrases follow-up questions before searching
- **MMR Diversity** — Retrieved chunks are non-redundant (λ = 0.7)
- **Source Citations** — Every answer shows the page number and chunk preview it was drawn from
- **Live Comparison** — Toggle hybrid vs. semantic-only to see the retrieval difference on the same query
- **Analytics Panel** — Chunk count, page count, word estimate per document in the sidebar
- **Graceful Errors** — Handles image-based / scanned PDFs with a clear user message instead of a crash

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/dev-mzeeshan/docsense-hybrid-rag.git
cd docsense-hybrid-rag

pip install -r requirements.txt
```

### 2. Set your API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

### 3. Run

```bash
streamlit run main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

**1. Upload PDFs**

Drag one or multiple PDFs into the upload area. DocSense processes each file, splits it into overlapping chunks, generates embeddings, and merges them into a shared FAISS index. Large documents (~50 pages) typically process in under 30 seconds.

**2. Choose retrieval mode**

Use the sidebar toggle to switch between:
- **Hybrid (recommended)** — BM25 + FAISS with MMR
- **Semantic only** — pure FAISS vector search

Try the same query in both modes to observe the difference on precision-sensitive questions.

**3. Ask questions**

The system maintains full conversation history. Follow-up questions like *"Tell me more about that"* or *"What does the previous clause say?"* work correctly because the retriever rephrases them before searching.

---

## Configuration

| Setting | Default | Description |
|---|---|---|
| `Model` | `llama-3.3-70b-versatile` | Groq model. 70b is more accurate; 8b is faster |
| `Hybrid Search` | `ON` | Toggle BM25 + FAISS vs. pure semantic |
| `Temperature` | `0.2` | Lower = more factual. Raise for exploratory queries |
| `Chunk Size` | `1000` | Characters per chunk. Smaller = more precise; larger = more context |
| `BM25 Weight` | `0.35` | Keyword component weight in the ensemble |
| `FAISS Weight` | `0.65` | Semantic component weight in the ensemble |

---

## Technical Deep-Dive

### Hybrid Retrieval — Why Not Just Better Embeddings?

A natural question: *why not just use a better embedding model instead of adding BM25?*

Better embeddings (e.g., `bge-large`, `text-embedding-3-large`) reduce the semantic search gap but do not eliminate it. BM25 is not a worse version of semantic search — it is a fundamentally different algorithm that excels in different conditions:

| Condition | BM25 | FAISS (cosine) |
|---|---|---|
| Exact term in query matches doc | ✅ High recall | ⚠️ Depends on embedding alignment |
| Paraphrased / synonymous query | ❌ Misses | ✅ Strong |
| Rare domain terms (medical, legal) | ✅ Term frequency | ⚠️ May drift |
| Conceptual / abstract question | ❌ No match | ✅ Strong |

Hybrid via Reciprocal Rank Fusion consistently outperforms either approach alone on domain-specific corpora. See: [Formal et al., 2021 — SPLADE](https://arxiv.org/abs/2107.05720).

### MMR vs. Top-K Similarity

Standard `similarity_search(k=5)` returns the 5 most similar chunks — which are often near-duplicates from the same paragraph. MMR (Maximum Marginal Relevance) penalises redundancy:

```
MMR score = λ · sim(query, chunk) − (1−λ) · max(sim(chunk, selected))
```

With `λ = 0.7`, the retriever balances relevance (70%) and diversity (30%), ensuring the LLM receives 5 genuinely different pieces of evidence rather than 5 copies of the same sentence.

### History-Aware Retrieval

Without history awareness:

```
User: "What are the payment terms?"
Bot:  [retrieves payment clause, answers correctly]
User: "When does the first payment occur?"
Bot:  [retrieves random chunks — "first" has no context]
```

With `create_history_aware_retriever`:

```
User: "When does the first payment occur?"
      ↓ (reformulated using chat history)
      "When does the first payment occur according to the payment terms in the contract?"
      ↓
      [retrieves correct clause]
```

The rephrasing step adds one LLM call but eliminates the most common multi-turn RAG failure mode.

---

## Project Structure

```
docsense-hybrid-rag/
│
├── main.py              # Streamlit UI — upload, chat, source viewer
├── rag_pipeline.py      # Core RAG logic — hybrid retriever, QA chain, analytics
├── requirements.txt
├── .env.example
└── README.md
```

---

## Stack

| Component | Technology | Purpose |
|---|---|---|
| LLM | Groq API (Llama 3.3 70B) | Fast inference, free tier available |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Local, no API cost |
| Vector Store | FAISS (CPU) | Fast dense retrieval + merge support |
| Keyword Search | BM25 (`rank_bm25`) | Exact-term matching |
| Ensemble | LangChain `EnsembleRetriever` | Reciprocal Rank Fusion |
| PDF Loading | PyPDFium2 | Reliable text extraction |
| UI | Streamlit | Rapid prototyping with good defaults |
| Orchestration | LangChain | Chain composition + history awareness |

---

## Requirements

```txt
streamlit
langchain
langchain-community
langchain-groq
langchain-huggingface
langchain-core
faiss-cpu
sentence-transformers
rank_bm25
pypdfium2
python-dotenv
```

---

## Roadmap

- [ ] Re-ranking with cross-encoder (`ms-marco-MiniLM-L-6-v2`)
- [ ] Document comparison mode (diff two PDFs)
- [ ] Export chat + citations as PDF report
- [ ] Urdu / Roman Urdu query support
- [ ] REST API endpoint (FastAPI wrapper)

---

## Author

**Muhammad Zeeshan** — AI Engineer & Co-Founder [@ChatSetGo](https://www.chatsetgo.tech)

[![Portfolio](https://img.shields.io/badge/Portfolio-dev--zeeshan--portfolio.vercel.app-0ea5e9?style=flat)](https://dev-zeeshan-portfolio.vercel.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-zeeshanofficial-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/zeeshanofficial)
[![GitHub](https://img.shields.io/badge/GitHub-dev--mzeeshan-181717?style=flat&logo=github)](https://github.com/dev-mzeeshan)

---

## License

MIT © 2026 Muhammad Zeeshan