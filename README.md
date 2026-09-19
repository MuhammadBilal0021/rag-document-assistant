# 🤖 RAG AI Assistant — Production-Ready Retrieval-Augmented Generation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade **Retrieval-Augmented Generation (RAG)** system that ingests documents (PDF, DOCX, web URLs), stores them as vector embeddings, and answers user questions with grounded citations — all exposed through a REST API and containerized with Docker.

> Built to demonstrate end-to-end AI engineering skills: LLMs, RAG pipelines, vector databases, REST APIs, and cloud-ready deployment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Evaluation](#evaluation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Overview

Traditional LLMs hallucinate because they generate answers from parametric memory alone. This system solves that by **grounding every response in retrieved source documents**, dramatically reducing hallucination and making answers verifiable.

**What it does:**

1. You upload a PDF or provide a URL
2. The system chunks, embeds, and indexes the document
3. When you ask a question, the most relevant chunks are retrieved from the vector store
4. Those chunks are injected into the LLM prompt as context
5. The LLM generates an answer — with citations back to the source

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                   │
│                                                             │
│  Document (PDF/DOCX/URL)                                    │
│       │                                                     │
│       ▼                                                     │
│  Text Extraction  ──►  Chunking  ──►  Embedding  ──►  FAISS│
│  (PyMuPDF/BS4)        (LangChain)    (OpenAI/OSS)   Index  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                      │
│                                                             │
│  User Question                                              │
│       │                                                     │
│       ▼                                                     │
│  Embed Query  ──►  FAISS Search  ──►  Top-k Chunks          │
│                                           │                 │
│                                           ▼                 │
│                                    Prompt Template          │
│                                           │                 │
│                                           ▼                 │
│                                    LLM (Groq/OpenAI)        │
│                                           │                 │
│                                           ▼                 │
│                                 Answer + Source Citations   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                          API LAYER (FastAPI)                │
│                                                             │
│   POST /upload  ──►  Ingestion Pipeline                     │
│   POST /query   ──►  Query Pipeline  ──►  JSON Response     │
│   GET  /health  ──►  Status Check                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **API Framework** | FastAPI |
| **LLM Provider** | Groq (Llama 3.3) / OpenAI GPT-4 / Anthropic Claude |
| **Embeddings** | OpenAI `text-embedding-3-small` / `sentence-transformers` |
| **Vector Database** | FAISS (local) → Pinecone (cloud, optional) |
| **Document Parsing** | PyMuPDF, python-docx, BeautifulSoup4 |
| **Data Processing** | Pandas, NumPy |
| **Containerization** | Docker, Docker Compose |
| **Experiment Tracking** | MLflow / Weights & Biases (optional) |

---

## Features

### Core (MVP)

- **Multi-format ingestion** — Upload PDF files; web URL and DOCX support in progress
- **Smart chunking** — Configurable chunk size and overlap to preserve context boundaries
- **Vector search** — FAISS-powered semantic similarity search over embedded chunks
- **Grounded answers** — Every response includes source citations (document name + chunk reference)
- **REST API** — Clean FastAPI endpoints with automatic OpenAPI docs at `/docs`

### Advanced

- **Conversation memory** — Multi-turn context window so follow-up questions work naturally
- **Guardrails** — Input validation to filter unsafe or off-topic prompts before hitting the LLM
- **Re-ranking** — Cross-encoder re-ranking of retrieved chunks for higher answer precision
- **Evaluation metrics** — Tracks answer relevance, faithfulness, and retrieval precision
- **Streaming responses** — Token-by-token streaming via Server-Sent Events
- **Structured logging** — Request/response logging with latency tracking

---

## Project Structure

```
rag-ai-assistant/
│
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   ├── routes/
│   │   │   ├── upload.py        # POST /upload endpoint
│   │   │   └── query.py         # POST /query endpoint
│   │   └── models.py            # Pydantic request/response schemas
│   │
│   ├── core/
│   │   ├── config.py            # Settings (env vars, LLM config)
│   │   └── logging.py           # Structured logger setup
│   │
│   ├── ingestion/
│   │   ├── loader.py            # PDF / DOCX / URL document loaders
│   │   ├── chunker.py           # Text splitting strategies
│   │   └── embedder.py          # Embedding generation
│   │
│   ├── retrieval/
│   │   ├── vector_store.py      # FAISS index (build / query / persist)
│   │   └── reranker.py          # Cross-encoder re-ranking (advanced)
│   │
│   └── generation/
│       ├── llm_client.py        # LLM provider abstraction (Groq/OpenAI)
│       ├── prompt_templates.py  # System + RAG prompt templates
│       └── memory.py            # Conversation history manager
│
├── evals/
│   ├── run_evals.py             # Run answer relevance + faithfulness evals
│   ├── metrics.py               # Scoring functions
│   ├── cases.yaml               # 34 evaluation cases (factual, multihop, injection, etc.)
│   └── config.yaml              # Thresholds and configuration
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── notebooks/
│   └── rag_exploration.ipynb    # Prototyping and chunk quality analysis
│
├── .env.example                 # Environment variable template
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An API key from [Groq](https://console.groq.com), [OpenAI](https://platform.openai.com), or [Anthropic](https://console.anthropic.com)
- Docker (optional, for containerized deployment)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-ai-assistant.git
cd rag-ai-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
# LLM Provider (choose one)
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key

# LLM Model
LLM_MODEL=llama-3.3-70b-versatile   # or gpt-4o

# Embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai            # or local

# Vector Store
VECTOR_STORE_PATH=./data/faiss_index

# Chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
```

### 5. Run the development server

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## API Reference

### `POST /upload`

Upload and index a document.

**Request** (`multipart/form-data`)

| Field | Type | Description |
|---|---|---|
| `file` | `File` | PDF or DOCX file to ingest |

**Response**

```json
{
  "status": "success",
  "document_id": "a3f9c1d2",
  "chunks_indexed": 42,
  "message": "Document ingested and indexed successfully."
}
```

---

### `POST /query`

Ask a question against indexed documents.

**Request** (`application/json`)

```json
{
  "question": "What are the key findings of the report?",
  "top_k": 5,
  "session_id": "optional-for-memory"
}
```

**Response**

```json
{
  "answer": "The report identifies three key findings: ...",
  "sources": [
    {
      "document": "annual_report_2024.pdf",
      "chunk_id": 12,
      "excerpt": "The primary finding indicates that..."
    }
  ],
  "latency_ms": 843
}
```

---

### `GET /health`

Check service status.

```json
{
  "status": "healthy",
  "vector_store": "loaded",
  "documents_indexed": 3
}
```

---

## Configuration

All configuration is driven by environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama-3.3-70b-versatile` | LLM model to use for generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for generating embeddings |
| `CHUNK_SIZE` | `500` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `VECTOR_STORE_PATH` | `./data/faiss_index` | Path to persist FAISS index |

---

## Docker Deployment

### Build and run with Docker Compose

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### Build manually

```bash
docker build -t rag-assistant -f docker/Dockerfile .
docker run -p 8000:8000 --env-file .env rag-assistant
```

### `docker-compose.yml` overview

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data     # Persist FAISS index across restarts
    env_file:
      - .env
```

---

## Evaluation

The `evals/` module measures answer quality across three axes:

| Metric | What it measures |
|---|---|
| **Answer Relevance** | Does the answer address the question? |
| **Faithfulness** | Is the answer grounded in the retrieved context (no hallucinations)? |
| **Retrieval Precision** | Are the retrieved chunks actually relevant to the question? |

The test suite runs against 34 cases covering 6 categories: factual retrieval, multi-hop reasoning, contradiction handling, unanswerable queries, context overflow, and adversarial injection.

Run evals against a test set:

```bash
python evals/run_evals.py --mock --no-judge
```

---

## Roadmap

- [x] PDF ingestion pipeline
- [x] FAISS vector store
- [x] RAG query pipeline
- [x] FastAPI REST endpoints
- [x] Docker containerization
- [ ] DOCX + web URL ingestion
- [ ] Streaming responses (SSE)
- [ ] Pinecone cloud vector store integration
- [ ] Frontend UI (Streamlit)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] MLflow experiment tracking

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/streaming-responses`)
3. Commit your changes (`git commit -m 'Add streaming response support'`)
4. Push to the branch (`git push origin feature/streaming-responses`)
5. Open a Pull Request

---

## License

[MIT](LICENSE)

---

<p align="center">Built by Muhammad · AI Engineer · Islamabad, PK</p>
