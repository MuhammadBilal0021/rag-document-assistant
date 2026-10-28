import asyncio
import json
import secrets
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException
from starlette.responses import JSONResponse
from app.config import Settings
from app.api.routes import create_router

class BodyLimit:
    def __init__(self,app,limit):
        self.app=app;self.limit=limit
    async def __call__(self,scope,receive,send):
        if scope["type"]!="http" or scope["method"] not in {"POST","PUT","PATCH"}:
            return await self.app(scope,receive,send)
        messages=[];size=0
        try:
            async with asyncio.timeout(30):
                while True:
                    message=await receive()
                    if message["type"]=="http.disconnect": return
                    size+=len(message.get("body",b""))
                    if size>self.limit:
                        return await JSONResponse({"detail":"Request too large"},status_code=413)(scope,receive,send)
                    messages.append(message)
                    if not message.get("more_body",False): break
        except TimeoutError:
            return await JSONResponse({"detail":"Upload timed out"},status_code=408)(scope,receive,send)
        async def replay():
            return messages.pop(0) if messages else await receive()
        await self.app(scope,replay,send)

def create_app(settings=None,pipeline_factory=None):
    settings=settings or Settings()
    users=json.loads(settings.rag_users_json)
    if not isinstance(users,dict) or any(not isinstance(k,str) or len(k)<32 or not isinstance(v,str) or not v or len(v)>100 for k,v in users.items()):
        raise ValueError("RAG_USERS_JSON must map random tokens of at least 32 characters to owner IDs")
    Path(settings.data_dir).mkdir(parents=True,exist_ok=True)
    @asynccontextmanager
    async def lifespan(app):
        if pipeline_factory:
            app.state.pipeline=pipeline_factory(settings)
        elif users and settings.groq_api_key:
            from app.rag.pipeline import RAGPipeline
            app.state.pipeline=RAGPipeline(settings)
        yield
    app=FastAPI(title="Private PDF RAG Assistant",lifespan=lifespan)
    app.state.settings=settings
    app.state.slots=threading.BoundedSemaphore(2)
    app.state.pipeline=None
    def authorize(authorization:str=Header(default="")):
        for token,owner in users.items():
            if secrets.compare_digest(authorization.encode(),("Bearer "+token).encode()):
                if app.state.pipeline is None:
                    raise HTTPException(503,"Configure generation credentials before use")
                return owner
        raise HTTPException(401,"Valid access token required")
    app.include_router(create_router(authorize),prefix="/api")
    app.add_middleware(BodyLimit,limit=settings.max_upload_bytes+1024*1024)
    @app.get("/health")
    def health():
        return {"status":"ready" if app.state.pipeline is not None else "not_configured",
                "configured":bool(users and settings.groq_api_key)}
    return app

app=create_app()
