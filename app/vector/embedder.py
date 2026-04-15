# app/vector/embedder.py
from sentence_transformers import SentenceTransformer
from app.config import settings
import numpy as np

class Embedder:
    def __init__(self):
        print(f"Loading embedding model: {settings.embed_model}")
        self.model = SentenceTransformer(settings.embed_model)
        print("Embedding model loaded.")

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Takes chunks from chunker.py and adds an 'embedding' key to each.
        Returns the same chunks with embeddings added.
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single search query.
        Called at query time, not ingestion time.
        """
        return self.model.encode(query)