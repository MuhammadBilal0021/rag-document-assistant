import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
import faiss
import numpy as np

class FAISSVectorStore:
    # SQLite is authoritative; vectors and metadata commit together.
    # FAISS is rebuilt from the owner's bounded collection for each search.
    def __init__(self,settings,dimension,model_name):
        self.settings=settings
        self.dimension=dimension
        root=Path(settings.data_dir).resolve()
        root.mkdir(parents=True,exist_ok=True)
        self.path=root/"documents.sqlite3"
        with closing(self._connect()) as conn,conn:
            conn.executescript("CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY,value TEXT);"
                "CREATE TABLE IF NOT EXISTS documents(owner TEXT,id TEXT,name TEXT,hash TEXT,created REAL,PRIMARY KEY(owner,id),UNIQUE(owner,hash));"
                "CREATE TABLE IF NOT EXISTS chunks(owner TEXT,doc_id TEXT,text TEXT,metadata TEXT,embedding BLOB);"
                "CREATE INDEX IF NOT EXISTS chunk_owner ON chunks(owner);"
                "CREATE TABLE IF NOT EXISTS requests(owner TEXT,created REAL);")
            expected=json.dumps([model_name,dimension])
            row=conn.execute("SELECT value FROM config WHERE key='embedding'").fetchone()
            if row and row[0]!=expected:
                raise ValueError("Embedding model changed. Use a new data directory and re-upload documents.")
            conn.execute("INSERT OR IGNORE INTO config VALUES('embedding',?)",(expected,))
    def _connect(self):
        return sqlite3.connect(self.path,timeout=10)
    def _expire(self,conn):
        cutoff=time.time()-self.settings.retention_days*86400
        conn.execute("DELETE FROM chunks WHERE EXISTS (SELECT 1 FROM documents d WHERE d.owner=chunks.owner AND d.id=chunks.doc_id AND d.created < ?)",(cutoff,))
        conn.execute("DELETE FROM documents WHERE created < ?",(cutoff,))
    def consume_request(self,owner):
        with closing(self._connect()) as conn,conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM requests WHERE created < ?",(time.time()-3600,))
            count=conn.execute("SELECT count(*) FROM requests WHERE owner=?",(owner,)).fetchone()[0]
            if count>=self.settings.requests_per_hour:
                raise PermissionError("Hourly request quota exceeded")
            conn.execute("INSERT INTO requests VALUES(?,?)",(owner,time.time()))
    def add_document(self,owner,doc_id,name,digest,chunks):
        if not chunks or len(chunks)>self.settings.max_chunks:
            raise ValueError("Invalid chunk count")
        rows=[]
        for c in chunks:
            vector=np.asarray(c["embedding"],dtype="float32")
            if vector.shape!=(self.dimension,) or not np.isfinite(vector).all():
                raise ValueError("Invalid embedding")
            rows.append((owner,doc_id,c["text"],json.dumps(c["metadata"]),vector.tobytes()))
        with closing(self._connect()) as conn,conn:
            conn.execute("BEGIN IMMEDIATE")
            self._expire(conn)
            existing=conn.execute("SELECT id FROM documents WHERE owner=? AND hash=?",(owner,digest)).fetchone()
            if existing:
                return existing[0]
            # Replacing the same visible filename is atomic.
            previous=conn.execute("SELECT id FROM documents WHERE owner=? AND name=?",(owner,name)).fetchall()
            for old in previous:
                conn.execute("DELETE FROM chunks WHERE owner=? AND doc_id=?",(owner,old[0]))
                conn.execute("DELETE FROM documents WHERE owner=? AND id=?",(owner,old[0]))
            count=conn.execute("SELECT count(*) FROM documents WHERE owner=?",(owner,)).fetchone()[0]
            if count>=self.settings.max_documents_per_owner:
                raise ValueError("Document quota exceeded")
            conn.execute("INSERT INTO documents VALUES(?,?,?,?,?)",(owner,doc_id,name,digest,time.time()))
            conn.executemany("INSERT INTO chunks VALUES(?,?,?,?,?)",rows)
        return doc_id
    def similarity_search(self,owner,query_embedding,k=5):
        with closing(self._connect()) as conn,conn:
            self._expire(conn)
            rows=conn.execute("SELECT text,metadata,embedding FROM chunks WHERE owner=?",(owner,)).fetchall()
        if not rows:
            return []
        vectors=np.stack([np.frombuffer(r[2],dtype="float32") for r in rows])
        index=faiss.IndexFlatIP(self.dimension)
        index.add(vectors)
        query=np.asarray(query_embedding,dtype="float32").reshape(1,-1)
        if query.shape[1]!=self.dimension:
            raise ValueError("Embedding dimension mismatch")
        scores,ids=index.search(query,min(k,len(rows)))
        return [{"text":rows[i][0],"metadata":json.loads(rows[i][1]),"score":float(score)}
                for score,i in zip(scores[0],ids[0]) if i>=0]
    def list_documents(self,owner):
        with closing(self._connect()) as conn,conn:
            self._expire(conn)
            return [{"id":r[0],"name":r[1],"created":r[2]} for r in conn.execute("SELECT id,name,created FROM documents WHERE owner=? ORDER BY created DESC",(owner,))]
    def delete_by_doc_id(self,owner,doc_id):
        with closing(self._connect()) as conn,conn:
            cursor=conn.execute("DELETE FROM documents WHERE owner=? AND id=?",(owner,doc_id))
            conn.execute("DELETE FROM chunks WHERE owner=? AND doc_id=?",(owner,doc_id))
            return cursor.rowcount>0
