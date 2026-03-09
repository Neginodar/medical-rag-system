# 🏥 Medical RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline for medical question answering, built on top of the [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) dataset. The system supports three interchangeable vector stores, cross-encoder re-ranking, LLM-powered generation, a REST API, and an interactive Gradio UI — all runnable on a Kaggle GPU notebook.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Dataset** | MedQuAD (lavita/MedQuAD) — 16,000+ medical Q&A pairs |
| **Embeddings** | `BAAI/bge-base-en-v1.5` via `sentence-transformers` |
| **Vector Stores** | Pinecone (cloud), FAISS (local), ChromaDB (local) |
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | Groq API — `llama-3.3-70b-versatile` |
| **Evaluation** | Custom embedding-based metrics (Answer Relevancy, Context Coverage, Ground Truth Similarity) |
| **API** | FastAPI with `/ask` and `/compare` endpoints |
| **UI** | Gradio interface with retriever selector and re-ranking toggle |
| **Platform** | Kaggle (NVIDIA Tesla T4 GPU) |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Retrieval Layer               │
│  ┌─────────┐  ┌───────┐  ┌──────────┐  │
│  │Pinecone │  │ FAISS │  │ChromaDB  │  │
│  └─────────┘  └───────┘  └──────────┘  │
└─────────────────────────────────────────┘
    │  top-10 candidates
    ▼
┌─────────────────────────────────────────┐
│         Cross-Encoder Re-ranking        │
│     (ms-marco-MiniLM-L-6-v2)           │
│           top-5 documents               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│          Generation (Groq API)          │
│       llama-3.3-70b-versatile           │
│  Structured: Summary + Evidence +       │
│              Clinical Context           │
└─────────────────────────────────────────┘
    │
    ▼
  Answer
```

---

## 📂 Repository Structure

```
medical-rag-system/
├── medical-multimodal-rag.ipynb   # Main notebook (full pipeline)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.12+
- A [Pinecone](https://www.pinecone.io/) API key (free tier works)
- A [Groq](https://console.groq.com/) API key (free tier works)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Keys

If running locally, create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_key_here
GROQ_API_KEY=your_groq_key_here
```

If running on **Kaggle**, add your keys via **Kaggle Secrets** (Add-ons → Secrets) with the names `PINECONE_API_KEY` and `GROQ_API_KEY`.

### 4. Run the Notebook

Open `medical-multimodal-rag.ipynb` and run all cells in order. The notebook will:

1. Install dependencies
2. Load the MedQuAD dataset (~16k Q&A pairs)
3. Embed and index documents into Pinecone, FAISS, and ChromaDB
4. Run the RAG pipeline
5. Evaluate results
6. Launch the FastAPI server and Gradio UI

---

## 🔬 Pipeline Details

### Data Ingestion

- Loads `lavita/MedQuAD` from HuggingFace Datasets
- Combines `question` + `answer` fields into a single text chunk per document
- Encodes all chunks using `BAAI/bge-base-en-v1.5` (768-dim embeddings)

### Vector Store Indexing

| Store | Index Type | Notes |
|---|---|---|
| **Pinecone** | Serverless (cosine) | Cloud-hosted, persistent |
| **FAISS** | `IndexFlatIP` (inner product) | In-memory, fast local search |
| **ChromaDB** | Default HNSW | Persistent local collection |

### Retrieval

Each retriever returns the top-10 most similar documents for a given query using cosine/inner-product similarity.

### Re-ranking

A `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) scores each (query, document) pair and reorders the top-10 results, keeping the top-5 for generation.

### Generation

The top-5 re-ranked documents are passed as context to `llama-3.3-70b-versatile` via the Groq API. The model is prompted to produce a structured answer with three sections:
- **Summary** — concise direct answer
- **Evidence** — citations from retrieved documents
- **Clinical Context** — broader clinical background

---

## 📊 Evaluation Results

Evaluation was performed on 5 representative medical queries using embedding-based metrics computed with `sentence-transformers`:

| Metric | Description | Score |
|---|---|---|
| **Answer Relevancy** | Cosine similarity between answer and query embeddings | **0.830** |
| **Context Coverage** | Max cosine similarity between answer and retrieved context | **0.830** |
| **Ground Truth Similarity** | Cosine similarity between answer and matched MedQuAD ground truth | **0.715** |

### Per-Query Results

| Query | Relevancy | Coverage | GT Similarity |
|---|---|---|---|
| Effects of metformin on type 2 diabetes | 0.831 | 0.885 | 0.677 |
| Aspirin and cardiovascular disease | 0.829 | 0.786 | 0.755 |
| Role of vitamin D in bone health | 0.869 | 0.860 | 0.838 |
| Exercise and depression symptoms | 0.859 | 0.751 | 0.610 |
| Side effects of statins | 0.761 | 0.867 | 0.693 |

---

## 🌐 REST API

The notebook launches a FastAPI server at `http://localhost:8000`.

### Endpoints

#### `GET /`
Health check.
```json
{ "status": "Medical RAG API running ✅" }
```

#### `POST /ask`
Ask a medical question.

**Request body:**
```json
{
  "query": "What are the effects of metformin on type 2 diabetes?",
  "retriever": "pinecone",
  "use_reranking": true
}
```

**Response:**
```json
{
  "query": "...",
  "answer": "...",
  "top_docs": [...],
  "retriever": "pinecone",
  "reranking": true
}
```

#### `GET /compare?query=...`
Retrieves top-3 results from all three vector stores simultaneously for side-by-side comparison.

Interactive docs available at `http://localhost:8000/docs`.

---

## 🖥️ Gradio UI

The notebook also launches a Gradio interface (accessible via a public share link):

- **Text input** for medical questions
- **Radio selector** to choose between Pinecone, FAISS, or ChromaDB
- **Checkbox** to enable/disable cross-encoder re-ranking
- **Answer panel** showing the structured LLM response
- **Documents panel** showing the top-3 retrieved chunks with similarity scores

Example questions are pre-loaded for quick testing.

---

## 📦 Dependencies

| Category | Package | Version |
|---|---|---|
| Core ML | `sentence-transformers` | 3.4.1 |
| Core ML | `torch` | 2.6.0 |
| Core ML | `faiss-cpu` | 1.10.0 |
| Vector DB | `pinecone` | 6.0.0 |
| Vector DB | `chromadb` | 0.6.3 |
| LLM | `groq` | 0.23.1 |
| RAG | `langchain` | 0.3.21 |
| RAG | `ragas` | 0.2.14 |
| Data | `datasets` | 3.4.0 |
| API | `fastapi` | 0.115.12 |
| API | `uvicorn` | 0.34.0 |
| UI | `gradio` | 5.23.3 |

Full list in [`requirements.txt`](./requirements.txt).

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `PINECONE_API_KEY` | ✅ Yes | Pinecone serverless index API key |
| `GROQ_API_KEY` | ✅ Yes | Groq API key for LLM inference |

---

## 📝 Notes

- The notebook is optimized for **Kaggle** (GPU: NVIDIA Tesla T4, Python 3.12). It can be adapted to run on Google Colab or locally with minor changes to the secrets handling.
- FAISS and ChromaDB indexes are built in-memory/locally during notebook execution. Only Pinecone persists across sessions.
- The Groq free tier has rate limits. If you hit a `429 Too Many Requests` error during batch evaluation, add a small `time.sleep()` between requests.

---

## 📄 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
