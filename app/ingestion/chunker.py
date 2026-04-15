# app/ingestion/chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from app.config import settings

def chunk_document(text: str, source: str, doc_id: str) -> list[dict]:
    """
    Split raw text into overlapping chunks with metadata.
    Returns a list of dicts, each with 'text' and 'metadata'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,        # 500 characters per chunk
        chunk_overlap=settings.chunk_overlap,  # 50 chars overlap between chunks
        separators=["\n\n", "\n", ". ", " "]  # tries to split on paragraphs first
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": source,
                "doc_id": doc_id,
                "chunk_index": i,
                "char_count": len(chunk_text)
            }
        })

    return chunks