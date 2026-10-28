import json
import threading
import time
from collections import OrderedDict
from groq import Groq
from app.vector.store import FAISSVectorStore
from app.rag.prompts import SYSTEM_PROMPT

ABSTAIN="I cannot find this information in the uploaded documents."

def validate_answer(raw,results):
    try:
        data=json.loads(raw)
        answer=data["answer"]
        citations=data["citations"]
        if not isinstance(answer,str) or not answer.strip() or len(answer)>8000 or not isinstance(citations,list):
            raise ValueError("Invalid response")
        if not citations:
            return {"answer":ABSTAIN,"sources":[]}
        sources=[]
        seen=set()
        for citation in citations:
            sid=citation["source_id"]
            quote=citation["quote"]
            if sid not in results or not isinstance(quote,str) or len(quote)<12 or quote not in results[sid]["text"]:
                raise ValueError("Unsupported citation")
            r=results[sid]
            if (sid,quote) not in seen:
                sources.append({"source_id":sid,"quote":quote,"source":r["metadata"]["source"],
                    "document_id":r["metadata"]["doc_id"],"page":r["metadata"]["page"],
                    "chunk_index":r["metadata"]["chunk_index"],"score":round(r["score"],4)})
                seen.add((sid,quote))
        return {"answer":answer,"sources":sources}
    except (ValueError,KeyError,TypeError):
        return {"answer":ABSTAIN,"sources":[]}
    # Exact quotes prove provenance, not that every generated claim follows from them.

class RAGPipeline:
    def __init__(self,settings):
        self.settings=settings
        from app.vector.embedder import Embedder
        self.embedder=Embedder(settings.embed_model)
        self.store=FAISSVectorStore(settings,self.embedder.dimension,settings.embed_model)
        self.client=Groq(api_key=settings.groq_api_key,timeout=20,max_retries=0)
        self.history=OrderedDict()
        self.lock=threading.RLock()
    def get_history(self,owner,session):
        with self.lock:
            entry=self.history.get((owner,session))
            if not entry or entry[0]<time.time()-86400:
                self.history.pop((owner,session),None)
                return []
            return list(entry[1])
    def clear_history(self,owner,session):
        with self.lock:
            self.history.pop((owner,session),None)
    def clear_owner_history(self,owner):
        with self.lock:
            for key in list(self.history):
                if key[0]==owner: self.history.pop(key,None)
    def query(self,owner,question,session_id):
        previous=self.get_history(owner,session_id)
        # Include the previous question as retrieval context for short follow-ups.
        retrieval=question if not previous else previous[-1]["question"]+"\nFollow-up: "+question
        results=self.store.similarity_search(owner,self.embedder.embed_query(retrieval),self.settings.top_k)
        if not results:
            return {"answer":ABSTAIN,"sources":[],"session_id":session_id}
        refs={f"S{i+1}":r for i,r in enumerate(results)}
        response=self.client.chat.completions.create(model=self.settings.model_name,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":json.dumps({
                "question":question,"history":previous,
                "evidence":[{"source_id":sid,"text":r["text"]} for sid,r in refs.items()]})}],
            response_format={"type":"json_object"},temperature=0,max_tokens=1200)
        result=validate_answer(response.choices[0].message.content,refs)
        with self.lock:
            key=(owner,session_id)
            self.history[key]=(time.time(),(previous+[{"question":question,"answer":result["answer"]}])[-3:])
            self.history.move_to_end(key)
            while len(self.history)>1000: self.history.popitem(last=False)
        return {**result,"session_id":session_id}
