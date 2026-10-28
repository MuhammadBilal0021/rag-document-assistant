import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_document

class QueryRequest(BaseModel):
    question: str=Field(min_length=1,max_length=2000)
    session_id: str=Field(default_factory=lambda:str(uuid.uuid4()),pattern=r"^[a-zA-Z0-9_-]{1,64}$")

def create_router(authorize):
    router=APIRouter()
    def service(request):
        return request.app.state.pipeline
    def enter(request,owner):
        pipeline=service(request)
        if not request.app.state.slots.acquire(blocking=False):
            raise HTTPException(429,"Service busy; retry later")
        try:
            pipeline.store.consume_request(owner)
        except PermissionError as exc:
            request.app.state.slots.release()
            raise HTTPException(429,str(exc)) from exc
        except Exception:
            request.app.state.slots.release()
            raise

    @router.post("/upload")
    def upload(request:Request,file:UploadFile=File(...),owner:str=Depends(authorize)):
        name=(file.filename or "")
        if not name.lower().endswith(".pdf") or len(name)>120 or any(c in name for c in '/\\:') or any(ord(c)<32 for c in name):
            raise HTTPException(400,"Use a simple PDF filename without directory components")
        settings=request.app.state.settings
        enter(request,owner)
        temporary=None
        try:
            # Uploaded filenames are display metadata only, never filesystem paths.
            with tempfile.NamedTemporaryFile(suffix=".pdf",dir=settings.data_dir,delete=False) as target:
                temporary=Path(target.name)
                digest=hashlib.sha256()
                total=0
                while chunk:=file.file.read(65536):
                    total+=len(chunk)
                    if total>settings.max_upload_bytes:
                        raise HTTPException(413,"PDF exceeds upload limit")
                    target.write(chunk);digest.update(chunk)
            pages=load_pdf(temporary,settings.max_pages)
            doc_id=str(uuid.uuid4())
            chunks=chunk_document(pages,name,doc_id,settings)
            pipeline=service(request)
            embedded=pipeline.embedder.embed_chunks(chunks)
            with pipeline.lock:
                stored=pipeline.store.add_document(owner,doc_id,name,digest.hexdigest(),embedded)
                pipeline.clear_owner_history(owner)
            return {"document_id":stored,"message":"Document indexed","chunks_indexed":len(chunks)}
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(400,str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500,"Document processing failed") from exc
        finally:
            if temporary: temporary.unlink(missing_ok=True)
            file.file.close()
            request.app.state.slots.release()

    @router.post("/query")
    def query(request:Request,req:QueryRequest,owner:str=Depends(authorize)):
        if not req.question.strip():
            raise HTTPException(400,"Question cannot be empty")
        enter(request,owner)
        try:
            with service(request).lock:
                return service(request).query(owner,req.question,req.session_id)
        except Exception as exc:
            raise HTTPException(502,"Answer service unavailable; retry later") from exc
        finally:
            request.app.state.slots.release()

    @router.get("/documents")
    def documents(request:Request,owner:str=Depends(authorize)):
        return {"documents":service(request).store.list_documents(owner)}

    @router.delete("/documents/{doc_id}")
    def delete_document(doc_id:str,request:Request,owner:str=Depends(authorize)):
        with service(request).lock:
            if not service(request).store.delete_by_doc_id(owner,doc_id):
                raise HTTPException(404,"Document not found")
            service(request).clear_owner_history(owner)
        return {"deleted":True}

    @router.get("/sessions/{session_id}/history")
    def history(session_id:str,request:Request,owner:str=Depends(authorize)):
        return {"turns":service(request).get_history(owner,session_id)}

    @router.delete("/sessions/{session_id}/history")
    def clear_history(session_id:str,request:Request,owner:str=Depends(authorize)):
        service(request).clear_history(owner,session_id)
        return {"cleared":True}
    return router
