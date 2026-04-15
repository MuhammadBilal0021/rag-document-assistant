# app/vector/store.py
import faiss
import numpy as np
import json
import os
from app.config import settings

class FAISSVectorStore:
    def __init__(self):
        self.dimension = 384        # must match embedder output
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []          # stores chunk text + metadata per vector

    def add_documents(self, chunks: list[dict]):
        """
        Takes embedded chunks and adds them to the FAISS index.
        Each chunk must have 'embedding', 'text', and 'metadata' keys.
        """
        embeddings = np.array(
            [chunk["embedding"] for chunk in chunks]
        ).astype("float32")

        self.index.add(embeddings)

        for chunk in chunks:
            self.metadata.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })

        print(f"Added {len(chunks)} vectors. Total: {self.index.ntotal}")

    def similarity_search(self, query_embedding: np.ndarray, k: int = None) -> list[dict]:
        """
        Find top-k most similar chunks to a query embedding.
        Returns list of chunks with their similarity scores.
        """
        if k is None:
            k = settings.top_k

        query = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:           # FAISS returns -1 if not enough results
                continue
            result = self.metadata[idx].copy()
            result["score"] = float(dist)
            results.append(result)

        return results

    def save_index(self):
        """Persist FAISS index and metadata to disk."""
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.index, settings.faiss_index_path)
        with open(settings.metadata_path, "w") as f:
            json.dump(self.metadata, f)
        print(f"Index saved. {self.index.ntotal} vectors.")

    def load_index(self):
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(settings.faiss_index_path):
            print("No existing index found. Starting fresh.")
            return
        self.index = faiss.read_index(settings.faiss_index_path)
        with open(settings.metadata_path, "r") as f:
            self.metadata = json.load(f)
        print(f"Index loaded. {self.index.ntotal} vectors.")

    def delete_by_doc_id(self, doc_id: str):
        """Remove all chunks belonging to a document."""
        self.metadata = [
            m for m in self.metadata
            if m["metadata"]["doc_id"] != doc_id
        ]
        # Rebuild index from remaining vectors
        self.index = faiss.IndexFlatL2(self.dimension)
        print(f"Removed doc {doc_id}. Rebuilding index...")