# Private PDF RAG Assistant
Upload text PDFs, ask questions, and inspect exact source quotes with document names and page numbers. FastAPI serves the API, SentenceTransformers embeds locally, FAISS ranks each owner's vectors, and Groq generates answers.

A portfolio/reference implementation with bounded, authenticated document handling. It does not guarantee factual accuracy or production certification.

## Setup
Use Python 3.13. Clone https://github.com/MuhammadBilal0021/rag-document-assistant.git, enter the repository, create a virtual environment with python -m venv .venv, activate it, and run pip install -r requirements.txt.

Copy .env.example to .env. Set GROQ_API_KEY and configure RAG_USERS_JSON with a different random token per owner:

    RAG_USERS_JSON={"replace-with-a-random-token-of-at-least-32-characters":"alice"}

Generate real tokens using python -c "import secrets; print(secrets.token_urlsafe(32))". Keep them private. This is administrator-managed bearer access, not a registration/login system.

Run uvicorn app.main:app --host 127.0.0.1 --port 8000 and, separately, streamlit run frontend/app.py. The first configured startup downloads the embedding model. Groq receives the question, recent conversation and retrieved excerpts; upload only documents your provider policy permits. Indexed text and embeddings persist locally in DATA_DIR.

## Implemented features
- PDF text extraction with page metadata; scanned PDFs require external OCR.
- Configurable character chunks and normalized local embeddings.
- Owner-scoped retrieval and transactional SQLite storage of vectors and metadata. FAISS indexes are rebuilt from the owner's bounded collection per query.
- Same-name replacement, content-hash deduplication, owner-scoped listing/deletion and expiry.
- Structured generation with exact source-ID/quote validation; invalid evidence leads to abstention.
- Three-turn context scoped by owner/session. The previous question also provides retrieval context for follow-ups.
- Streamlit upload/chat/delete UI, health endpoint and offline regression tests.

Quote validation establishes provenance, not logical support for every answer claim. No confidence score is presented. Hybrid retrieval, reranking, OCR, DOCX/URL uploads and SSE streaming are not implemented.

## API
Authenticated routes require Authorization: Bearer <token>:
- POST /api/upload: multipart file; returns document_id and chunks_indexed.
- POST /api/query: question and optional session_id; returns answer, sources and session_id.
- GET /api/documents: your document metadata.
- DELETE /api/documents/{document_id}: delete your document/vectors.
- GET /api/sessions/{session_id}/history: your recent turns.
- DELETE /api/sessions/{session_id}/history: clear your turns.
- GET /health: public readiness/configuration status without secrets.

Interactive schemas are at /docs. Session IDs do not confer authority; the access token selects the owner.

## Limits and privacy
5 MB uploads, 100 pages, 500,000 extracted characters, 1,000 chunks/document, 25 documents/owner, two active expensive requests and 60 upload/query requests/owner/hour. Questions are limited to 2,000 characters. Request limits apply even without Content-Length. Parsing/embedding runs outside the API event loop, though hostile parser workloads may require stronger process isolation.

Uploads use random temporary paths and are removed after processing. Text/vectors expire after seven days by default, purged during document operations/search. In-memory history retains at most three turns/session and 1,000 sessions, with a one-day access expiry; restart clears it. Replacing/deleting a document clears its owner's chat context. Backups can retain deleted data: manage backup retention separately.

Use separate tokens per owner and protect DATA_DIR. A stolen token grants its owner's access. Run one worker/replica: history and concurrency controls are process-local. TLS, reverse-proxy protections and process isolation are deployment responsibilities.

## Configuration
See .env.example. Actual names: MODEL_NAME, EMBED_MODEL, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, REQUESTS_PER_HOUR, RETENTION_DAYS, RAG_USERS_JSON and GROQ_API_KEY. Chunk sizes measure characters. Changing embedding models/dimensions requires a new DATA_DIR and document re-upload.

## Docker and migration
Run docker compose up --build. Compose binds the API to localhost and persists data in a named volume; the container runs as a non-root user. Start Streamlit separately.

Old shared faiss.index/metadata.json files are not imported because they have no reliable owner information. Keep a backup and re-upload with the correct owner's token. Original document files and repository history are not deleted.

## Validation and roadmap
Run pip install -r requirements-dev.txt then python -m pytest -q. Tests use real PDF parsing, SQLite and FAISS with deterministic test embeddings and a mocked generator. They verify security/lifecycle behavior, not real retrieval quality or model accuracy. CI also checks dependency consistency and advisories.

Next: build human-reviewed question/evidence fixtures for one document domain; measure retrieval recall, citation correctness, faithfulness, abstention and latency; then compare reranking/hybrid retrieval to that baseline. See docs/verification.md for checked scope.
