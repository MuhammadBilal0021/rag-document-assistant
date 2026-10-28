import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name):
        self.model=SentenceTransformer(model_name,trust_remote_code=False)
        self.dimension=self.model.get_sentence_embedding_dimension()
    def embed_chunks(self,chunks):
        values=self.model.encode([c["text"] for c in chunks],normalize_embeddings=True)
        return [{**c,"embedding":values[i]} for i,c in enumerate(chunks)]
    def embed_query(self,query):
        return self.model.encode(query,normalize_embeddings=True)
