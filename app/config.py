from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env",extra="ignore")
    groq_api_key: str=""
    model_name: str="llama-3.3-70b-versatile"
    embed_model: str="all-MiniLM-L6-v2"
    data_dir: str="data"
    # JSON object: long random access token -> stable owner ID.
    rag_users_json: str="{}"
    chunk_size: int=Field(500,ge=100,le=2000)
    chunk_overlap: int=Field(50,ge=0,le=200)
    top_k: int=Field(5,ge=1,le=10)
    max_upload_bytes: int=5*1024*1024
    max_pages: int=100
    max_chunks: int=1000
    max_documents_per_owner: int=25
    requests_per_hour: int=60
    retention_days: int=7
