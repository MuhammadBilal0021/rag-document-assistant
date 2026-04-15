# app/rag/pipeline.py
from groq import Groq
from app.vector.embedder import Embedder
from app.vector.store import FAISSVectorStore
from app.rag.prompts import RAG_PROMPT
from app.config import settings

class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.store = FAISSVectorStore()
        self.store.load_index()
        self.client = Groq(api_key=settings.groq_api_key)
        self.history = {}           # session_id → list of turns

    def _get_history(self, session_id: str) -> str:
        """Get conversation history for a session as a string."""
        turns = self.history.get(session_id, [])
        if not turns:
            return "No previous conversation."
        return "\n".join([
            f"User: {t['question']}\nAssistant: {t['answer']}"
            for t in turns[-3:]     # last 3 turns only
        ])

    def _save_turn(self, session_id: str, question: str, answer: str):
        """Save a Q&A turn to session history."""
        if session_id not in self.history:
            self.history[session_id] = []
        self.history[session_id].append({
            "question": question,
            "answer": answer
        })

    def _format_sources(self, results: list[dict]) -> list[dict]:
        """Format retrieved chunks into clean citation objects."""
        sources = []
        for r in results:
            sources.append({
                "text": r["text"][:200],
                "source": r["metadata"]["source"],
                "chunk_index": r["metadata"]["chunk_index"],
                "score": round(r["score"], 4)
            })
        return sources

    def query(self, question: str, session_id: str = "default") -> dict:
        """
        Main method — takes a question, returns answer + citations.
        """
        # 1. Embed the question
        query_vec = self.embedder.embed_query(question)

        # 2. Retrieve top-k relevant chunks
        results = self.store.similarity_search(query_vec, k=settings.top_k)

        if not results:
            return {
                "answer": "No documents found. Please upload a document first.",
                "sources": [],
                "session_id": session_id
            }

        # 3. Build context from retrieved chunks
        context = "\n\n---\n\n".join([r["text"] for r in results])

        # 4. Get conversation history
        history = self._get_history(session_id)

        # 5. Build the prompt
        prompt = RAG_PROMPT.format(
            context=context,
            history=history,
            question=question
        )

        # 6. Call Groq LLM
        response = self.client.chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,        # low = factual, consistent
            max_tokens=1024
        )

        answer = response.choices[0].message.content

        # 7. Save to history
        self._save_turn(session_id, question, answer)

        # 8. Return answer + citations
        return {
            "answer": answer,
            "sources": self._format_sources(results),
            "session_id": session_id
        }