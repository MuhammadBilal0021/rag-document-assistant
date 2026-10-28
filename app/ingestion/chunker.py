from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_document(pages, source, doc_id, settings):
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size")
    splitter=RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap)
    result=[]
    for page in pages:
        for text in splitter.split_text(page["text"]):
            result.append({"text":text,"metadata":{"source":source,"doc_id":doc_id,
                "page":page["page"],"chunk_index":len(result)}})
            if len(result)>settings.max_chunks:
                raise ValueError("Too many document chunks")
    return result
