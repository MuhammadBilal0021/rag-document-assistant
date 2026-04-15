# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    groq_api_key: str
    model_name: str = "llama-3.3-70b-versatile"

    # Embeddings
    embed_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 5

    # Paths
    faiss_index_path: str = "data/faiss.index"
    metadata_path: str = "data/metadata.json"

    class Config:
        env_file = ".env"

# Single instance — import this everywhere
settings = Settings()