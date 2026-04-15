from fastapi import APIRouter, UploadFile, File, HTTPException
from app.rag.pipeline import RAGPipeline
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_document
from pydantic import BaseModel
import shutil
import os

router = APIRouter()
pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported.")
    
    os.makedirs("data", exist_ok=True)
    save_path = f"data/{file.filename}"
    
    # Save file
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Re-index automatically
    text = load_document(save_path, "pdf")
    chunks = chunk_document(text, source=file.filename, doc_id=file.filename)
    
    # Reuse the pipeline's already-loaded embedder & store (avoids reloading the model)
    embedded_chunks = pipeline.embedder.embed_chunks(chunks)
    pipeline.store.add_documents(embedded_chunks)
    pipeline.store.save_index()
    
    return {"message": f"Uploaded and indexed: {file.filename}"}

@router.post("/query")
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = pipeline.query(req.question, session_id=req.session_id)
    return result

@router.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    history = pipeline.history.get(session_id, [])
    return {"session_id": session_id, "turns": history}