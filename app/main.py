# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting RAG Assistant...")
    print(f"Model: {settings.model_name}")
    print(f"Embed: {settings.embed_model}")
    yield
    print("Shutting down...")

app = FastAPI(
    title="RAG Assistant",
    description="Upload documents, ask questions, get cited answers.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")