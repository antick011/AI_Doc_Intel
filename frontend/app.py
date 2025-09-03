import sys
import os
import streamlit as st

# ✅ Ensure backend folder is discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.document_loader import save_uploaded_file, extract_text_from_pdf
from backend.text_splitter import split_text
from backend.vectorstore import VectorIndex
from backend.rag_chain import answer_question
from backend.summarizer import summarize

st.set_page_config(page_title="AI Document Intelligence", layout="wide")
st.title("📚 AI Document Intelligence – Enterprise Q&A")

with st.sidebar:
    st.header("Upload Documents")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    build_index = st.button("Build / Refresh Index")
    st.markdown("---")
    top_k = st.slider("Retriever top-k", 3, 10, 5)
    sum_mode = st.selectbox("Summary mode", ["short", "detailed"])

status = st.empty()

if uploads and build_index:
    paths = []
    for uf in uploads:
        path = save_uploaded_file(uf)
        paths.append(path)
    status.info("Extracting text & chunking…")
    all_chunks = []
    metas = []
    for p in paths:
        raw = extract_text_from_pdf(p)
        chunks = split_text(raw)
        all_chunks.extend(chunks)
        metas.extend([{"source": os.path.basename(p)}] * len(chunks))
    status.info("Creating vector index (FAISS)…")
    VectorIndex().create(all_chunks, metas)
    status.success(f"Indexed {len(all_chunks)} chunks from {len(paths)} document(s).")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a Question")
    q = st.text_input("Type your query about the uploaded documents")
    if st.button("Answer") and q:
        with st.spinner("Running RAG…"):
            answer, sources = answer_question(q, k=top_k)
        st.markdown("### Answer")
        st.write(answer)
        if sources:
            st.markdown("### Sources")
            for s in sources:
                st.caption(f"(Chunk {s['i']}) {s['doc']} – {s['preview']}")

with col2:
    st.subheader("Summarize a Document")
    sum_upload = st.file_uploader("Upload a PDF to summarize", type=["pdf"], key="sum")
    if st.button("Summarize") and sum_upload:
        path = save_uploaded_file(sum_upload)
        raw = extract_text_from_pdf(path)
        with st.spinner("Summarizing…"):
            summary = summarize(raw[:100000], mode=sum_mode)  # safety cutoff
        st.markdown("### Summary")
        st.write(summary)

st.markdown("---")
st.caption("Built with LangChain + FAISS. Switch providers in .env.")
