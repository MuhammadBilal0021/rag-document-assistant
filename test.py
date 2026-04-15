from app.rag.pipeline import RAGPipeline

import sys
sys.stdout.reconfigure(encoding='utf-8')

pipeline = RAGPipeline()

# Test 1 — factual question
result = pipeline.query(
    "What are Muhammad's technical skills?",
    session_id="test_session"
)
print("ANSWER:", result["answer"])
print("\nSOURCES:")
for s in result["sources"]:
    print(f"  - chunk {s['chunk_index']} | score {s['score']} | {s['text'][:100]}")

# Test 2 — follow up (tests memory)
result2 = pipeline.query(
    "What projects has he built?",
    session_id="test_session"
)
print("\n\nFOLLOW UP ANSWER:", result2["answer"])