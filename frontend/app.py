import streamlit as st
import requests
import uuid

# --- Config ---
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("📄 RAG Assistant")
    st.markdown("---")
    
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    
    if uploaded_file:
        if st.button("Upload & Index", type="primary"):
            with st.spinner("Uploading..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                response = requests.post(f"{API_URL}/api/upload", files=files)
                if response.status_code == 200:
                    st.success("✅ Document uploaded!")
                else:
                    st.error("❌ Upload failed.")

    st.markdown("---")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.caption(f"Session: {st.session_state.session_id[:8]}...")

# --- Main Chat Area ---
st.title("🤖 RAG Assistant")
st.caption("Ask questions about your uploaded documents.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for s in message["sources"]:
                    st.caption(f"Chunk {s['chunk_index']} | Score: {s['score']}")
                    st.caption(f"{s['text'][:200]}...")
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/query",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    
                    st.write(answer)
                    
                    if sources:
                        with st.expander("📚 Sources"):
                            for s in sources:
                                st.caption(f"Chunk {s['chunk_index']} | Score: {s['score']}")
                                st.caption(f"{s['text'][:200]}...")
                                st.markdown("---")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error("❌ API error. Is the server running?")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Run: uvicorn app.main:app --reload")