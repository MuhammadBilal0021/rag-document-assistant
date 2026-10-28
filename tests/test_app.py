import json
import threading
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import create_app
from app.rag.pipeline import RAGPipeline,validate_answer,ABSTAIN
from app.vector.store import FAISSVectorStore

TOKENS={"a"*40:"alice","b"*40:"bob"}
def headers(who="a"):
    return {"Authorization":"Bearer "+who*40}
def pdf(text):
    with pymupdf.open() as doc:
        page=doc.new_page();page.insert_text((72,72),text)
        return doc.tobytes()

class FakePipeline(RAGPipeline):
    # Exercises real storage, routing, history and citation validation; no remote model.
    def __init__(self,settings):
        self.settings=settings
        self.store=FAISSVectorStore(settings,2,"test-embedding")
        self.lock=threading.RLock();self.history=OrderedDict()
        self.embedder=SimpleNamespace(embed_chunks=lambda cs:[{**c,"embedding":np.array([1.,0.])} for c in cs],
            embed_query=lambda q:np.array([1.,0.]))
        def complete(**kwargs):
            evidence=json.loads(kwargs["messages"][1]["content"])["evidence"][0]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "answer":evidence["text"],"citations":[{"source_id":evidence["source_id"],"quote":evidence["text"]}]})))])
        self.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))

@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None,data_dir=str(tmp_path),groq_api_key="fake",
        rag_users_json=json.dumps(TOKENS),requests_per_hour=100)

@pytest.fixture
def client(settings):
    with TestClient(create_app(settings,FakePipeline)) as client:
        yield client

def upload(client,text="Private alpha content belongs only to Alice.",name="report.pdf",owner="a"):
    return client.post("/api/upload",files={"file":(name,pdf(text),"application/pdf")},headers=headers(owner))

def test_auth_and_health(client):
    assert client.get("/health").status_code==200
    assert client.get("/api/documents").status_code==401
    assert client.post("/api/query",json={"question":"hi"}).status_code==401

@pytest.mark.parametrize("name",["../escape.pdf",r"..\escape.pdf","C:escape.pdf","x.txt"])
def test_path_traversal_rejected(client,settings,name):
    assert upload(client,name=name).status_code==400
    assert not list(Path(settings.data_dir).glob("*.pdf"))

def test_bad_pdf_removed(client,settings):
    response=client.post("/api/upload",files={"file":("bad.pdf",b"not pdf")},headers=headers())
    assert response.status_code==400
    assert not list(Path(settings.data_dir).glob("*.pdf"))

def test_owner_isolation_and_history(client):
    assert upload(client).status_code==200
    alice=client.post("/api/query",headers=headers(),json={"question":"What is private?","session_id":"shared"}).json()
    bob=client.post("/api/query",headers=headers("b"),json={"question":"What is private?","session_id":"shared"}).json()
    assert "Alice" in alice["answer"] and alice["sources"][0]["page"]==1
    assert bob["sources"]==[] and bob["answer"]==ABSTAIN
    assert client.get("/api/sessions/shared/history",headers=headers("b")).json()["turns"]==[]

def test_replace_delete_and_restart(client,settings):
    old=upload(client).json()["document_id"]
    new=upload(client,text="Updated alpha document contains a new policy.").json()["document_id"]
    docs=client.get("/api/documents",headers=headers()).json()["documents"]
    assert len(docs)==1 and docs[0]["id"]==new and old!=new
    other=upload(client,text="Another independent document survives deletion.",name="other.pdf").json()["document_id"]
    assert client.delete("/api/documents/"+new,headers=headers("b")).status_code==404
    assert client.delete("/api/documents/"+new,headers=headers()).status_code==200
    restarted=FAISSVectorStore(settings,2,"test-embedding")
    assert [d["id"] for d in restarted.list_documents("alice")]==[other]
    results=restarted.similarity_search("alice",np.array([1.,0.]))
    assert results and all(r["metadata"]["doc_id"]==other for r in results)

def test_duplicate_upload_idempotent(client):
    content=pdf("Identical content should not create duplicate chunks.")
    def send():
        return client.post("/api/upload",headers=headers(),files={"file":("same.pdf",content)}).json()["document_id"]
    assert send()==send()
    assert len(client.get("/api/documents",headers=headers()).json()["documents"])==1

def test_model_change_rejected(settings):
    FAISSVectorStore(settings,2,"model-a")
    with pytest.raises(ValueError,match="Embedding model changed"):
        FAISSVectorStore(settings,3,"model-b")

def test_history_cleared_on_delete(client):
    did=upload(client).json()["document_id"]
    client.post("/api/query",headers=headers(),json={"question":"What?","session_id":"one"})
    client.delete("/api/documents/"+did,headers=headers())
    assert client.get("/api/sessions/one/history",headers=headers()).json()["turns"]==[]

def test_limits(settings):
    settings.max_upload_bytes=32
    settings.requests_per_hour=1
    with TestClient(create_app(settings,FakePipeline)) as c:
        assert upload(c).status_code==413
        assert c.post("/api/query",headers=headers(),json={"question":"hello"}).status_code==429
        assert c.post("/api/query",headers=headers(),json={"question":"x"*2001}).status_code==422
        assert c.post("/api/query",headers=headers(),content=b"x"*(1024*1024+33)).status_code==413

def test_abstains_on_fabricated_quote():
    refs={"S1":{"text":"The actual manual has twelve pages.","metadata":{"source":"manual.pdf","doc_id":"one","page":1,"chunk_index":0},"score":1}}
    raw=json.dumps({"answer":"Invented claim","citations":[{"source_id":"S1","quote":"fabricated quote not found"}]})
    assert validate_answer(raw,refs)=={"answer":ABSTAIN,"sources":[]}
    valid=json.dumps({"answer":"Twelve pages.","citations":[{"source_id":"S1","quote":"manual has twelve pages"}]})
    assert len(validate_answer(valid,refs)["sources"])==1
    assert validate_answer("not json",refs)["sources"]==[]

def test_empty_pdf_rejected(client):
    with pymupdf.open() as doc:
        doc.new_page();data=doc.tobytes()
    assert client.post("/api/upload",headers=headers(),files={"file":("empty.pdf",data)}).status_code==400

def test_document_limit_rolls_back(settings):
    settings.max_documents_per_owner=1
    store=FAISSVectorStore(settings,2,"test")
    chunks=[{"text":"test","metadata":{"doc_id":"a"},"embedding":np.array([1.,0.])}]
    store.add_document("alice","a","a.pdf","hash-a",chunks)
    with pytest.raises(ValueError,match="quota"):
        store.add_document("alice","b","b.pdf","hash-b",chunks)
    assert len(store.list_documents("alice"))==1
