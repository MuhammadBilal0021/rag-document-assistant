# app/api/schemas.py
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"

class SourceChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    session_id: str

class UploadResponse(BaseModel):
    message: str
    path: str
    note: str

class HealthResponse(BaseModel):
    status: str
    model: str
    embed_model: str
    vectors_indexed: int