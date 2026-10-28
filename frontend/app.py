import os
import uuid
import requests
import streamlit as st

st.set_page_config(page_title="Private PDF Assistant",page_icon="📄",layout="wide")
st.title("Private PDF Assistant")
base=os.environ.get("RAG_API_URL","http://127.0.0.1:8000")
token=st.sidebar.text_input("Access token",type="password")
if st.session_state.get("token")!=token:
    st.session_state.clear()
    st.session_state.update(token=token,session=str(uuid.uuid4()),messages=[])
if not token:
    st.info("Enter your access token. Documents are private to its owner and retained for seven days by default.")
    st.stop()
def api(method,path,**kwargs):
    response=requests.request(method,base+"/api"+path,headers={"Authorization":"Bearer "+token},timeout=60,**kwargs)
    response.raise_for_status()
    return response.json()
try:
    uploaded=st.sidebar.file_uploader("PDF (up to 5 MB; text PDFs only)",type=["pdf"])
    if uploaded and st.sidebar.button("Upload / replace"):
        api("POST","/upload",files={"file":(uploaded.name,uploaded,"application/pdf")})
        st.session_state.messages=[]
        st.sidebar.success("Indexed. Same filename replaces the previous version.")
    for doc in api("GET","/documents")["documents"]:
        st.sidebar.caption(doc["name"])
        if st.sidebar.button("Delete",key=doc["id"]):
            api("DELETE","/documents/"+doc["id"])
            st.session_state.messages=[]
            st.rerun()
    if st.sidebar.button("Clear chat"):
        api("DELETE",f"/sessions/{st.session_state.session}/history")
        st.session_state.messages=[]
        st.rerun()
    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.write(item["content"])
            for source in item.get("sources",[]):
                st.caption(f"{source['source']} — page {source['page']} [{source['source_id']}]")
                st.write(source["quote"])
    if question:=st.chat_input("Ask about your documents"):
        result=api("POST","/query",json={"question":question,"session_id":st.session_state.session})
        st.session_state.messages.extend([{"role":"user","content":question},
            {"role":"assistant","content":result["answer"],"sources":result["sources"]}])
        st.rerun()
except requests.RequestException as exc:
    status=getattr(exc.response,"status_code",None)
    st.error(f"Request failed ({status or 'connection/timeout'}). Check your token, server status, or processing limits.")
