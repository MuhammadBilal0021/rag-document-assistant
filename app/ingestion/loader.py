# app/ingestion/loader.py
import fitz                          # PyMuPDF
import docx
import requests
from bs4 import BeautifulSoup

def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file page by page."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def load_docx(file_path: str) -> str:
    """Extract text from a DOCX file paragraph by paragraph."""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text.strip()

def load_url(url: str) -> str:
    """Fetch a webpage and extract clean text, stripping HTML tags."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts and styles — we only want readable text
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    return soup.get_text(separator="\n").strip()

def load_document(source: str, source_type: str) -> str:
    """
    Main entry point. Call this from anywhere in the app.

    source_type: "pdf" | "docx" | "url"
    """
    if source_type == "pdf":
        return load_pdf(source)
    elif source_type == "docx":
        return load_docx(source)
    elif source_type == "url":
        return load_url(source)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")