import os
import io
import math
import time
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple
from openai import OpenAI
from pypdf import PdfReader


client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url="https://api.ai.it.cornell.edu/v1",
)

CHAT_MODEL = "meta.llama-3.2-1b-instruct"
EMBED_MODEL = "google.text-embedding"  

# ====== Tools ======
def read_file_content(file) -> str:
    """Read txt or pdf files as plain text."""
    filename = file.name.lower()
    if filename.endswith(".txt"):
        raw = file.read()
        try:
            return raw.decode("utf-8")
        except Exception:
            return raw.decode("latin-1", errors="ignore")

    elif filename.endswith(".pdf"):
        reader = PdfReader(file)
        text = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text.append(content)
        return "\n".join(text)

    else:
        return ""

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    Chunk by character length, with overlap, to ensure long texts are searchable.
    You can also replace it with a sentence-based chunker.
    """
    text = text.replace("\r\n", "\n")
    tokens = list(text)
    chunks = []
    i = 0
    while i < len(tokens):
        j = i + chunk_size
        chunk = "".join(tokens[i:j])
        chunks.append(chunk)
        if j >= len(tokens):
            break
        i = j - overlap
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    """Call the embeddings interface of the agent, return a numpy array of shape (n, d)"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=np.float32)

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate the cosine similarity matrix"""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T

def retrieve(query: str, index: Dict, top_k: int = 4) -> List[Tuple[int, str, float]]:
    """Use vector retrieval Top-k, return (chunk_id, chunk_text, score)"""
    qv = embed_texts([query])  # (1, d)
    sims = cosine_sim_matrix(qv, index["embeds"])  # (1, n)
    order = np.argsort(-sims[0])[:top_k]
    return [(int(i), index["chunks"][int(i)], float(sims[0, int(i)])) for i in order]

def build_prompt(query: str, retrieved: List[Tuple[int, str, float]]) -> List[Dict]:
    """
    Based on the retrieval block to construct prompts, the model is required to "only answer based on the content of the document; 
    if there is no relevant information in the document, then say 'I don't know'".
    """
    context_blocks = []
    for i, text, score in retrieved:
        context_blocks.append(f"[Chunk #{i} | score={score:.3f}]\n{text}")

    system_msg = (
        "You are a helpful assistant that answers ONLY using the provided document excerpts. "
        "If the answer is not contained in the excerpts, say 'I don't know based on the document.' "
        "Cite chunk numbers when helpful."
    )
    context = "\n\n".join(context_blocks)
    user_msg = (
        f"User question: {query}\n\n"
        f"Here are the most relevant document excerpts:\n\n{context}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

# ====== Streamlit UI ======
st.set_page_config(page_title="RAG Q&A", page_icon="📚", layout="wide")
st.title("📚 RAG System: Upload → Chunk → Retrieve → Chat")

with st.sidebar:
    st.subheader("RAG Parameters")
    chunk_size = st.slider("Chunk size (chars)", 300, 1500, 800, 50)
    overlap = st.slider("Chunk overlap (chars)", 0, 400, 120, 10)
    top_k = st.slider("Top-k retrieved", 1, 10, 4, 1)
    if st.button("Clear chat history"):
        st.session_state.pop("messages", None)

# Multiple document upload (.txt / .pdf)
uploaded_files = st.file_uploader("Upload .txt or .md files", type=("txt", "pdf"), accept_multiple_files=True)

# Conversation & Index Cache
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! Upload documents and ask a question."}]
if "index" not in st.session_state:
    st.session_state["index"] = None

# Build or update vector index
if uploaded_files:
    all_texts = []
    for f in uploaded_files:
        txt = read_file_content(f)
        chunks = chunk_text(txt, chunk_size=chunk_size, overlap=overlap)

        if len(chunks) > 800:
            chunks = chunks[:800]

        all_texts.extend([f"[{f.name}] {c}" for c in chunks])

    embeds = embed_texts(all_texts)  # (n, d)
    st.session_state["index"] = {"chunks": all_texts, "embeds": embeds}
    st.success(f"Indexed {len(all_texts)} chunks from {len(uploaded_files)} file(s).")

# Render historical messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Input questions
question = st.chat_input("Ask about the uploaded documents…", disabled=(st.session_state["index"] is None))

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # Retrieve
    with st.spinner("Retrieving…"):
        retrieved = retrieve(question, st.session_state["index"], top_k=top_k)

    # Display retrieved blocks (visible interpretability)
    with st.expander("🔎 Retrieved Chunks", expanded=False):
        for i, text, score in retrieved:
            st.markdown(f"**Chunk #{i} · score={score:.3f}**")
            st.code(text[:1200])

    # output
    with st.chat_message("assistant"):
        messages = build_prompt(question, retrieved)
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True
        )
        answer = st.write_stream(stream)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
