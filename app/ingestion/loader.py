from pathlib import Path
import pymupdf as fitz

def load_pdf(path, max_pages=100):
    if Path(path).read_bytes()[:5]!=b"%PDF-":
        raise ValueError("File is not a PDF")
    pages=[]
    total=0
    with fitz.open(path) as doc:
        if doc.needs_pass or not 1 <= len(doc) <= max_pages:
            raise ValueError("Encrypted, empty, or oversized PDF")
        for index,page in enumerate(doc):
            text=page.get_text().strip()
            total+=len(text)
            if total>500000:
                raise ValueError("PDF text exceeds processing limit")
            if text:
                pages.append({"page":index+1,"text":text})
    if not pages:
        raise ValueError("No readable text found; scanned PDFs require OCR")
    return pages

def load_document(source,source_type="pdf"):
    if source_type!="pdf":
        raise ValueError("Only PDF ingestion is supported")
    return load_pdf(source)
