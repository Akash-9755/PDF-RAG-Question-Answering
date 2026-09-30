"""
Streamlit UI for the PDF RAG (Retrieval-Augmented Generation) app.

Combines the logic from create_database.py and main.py into a single
interactive web app: upload a PDF, build a vector index from it, then
chat with it. Everything lives in the center column (no sidebar).

Run with:
    streamlit run app.py
"""

import asyncio
import concurrent.futures
import os
import platform
import sys
import tempfile
import time
import warnings

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore", message="Direct use of automatic function calling")

# Windows only: the default asyncio "Proactor" event loop logs a noisy
# (harmless) ConnectionResetError when an async network connection (e.g.
# Google's gRPC client) is torn down. Switching to the Selector loop avoids
# the spurious traceback in the terminal.
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

load_dotenv()

LLM_TIMEOUT_SECONDS = 45

# --------------------------------------------------------------------------
# Page config + light styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📚",
    layout="centered",
)

st.markdown(
    """
    <style>
        .main .block-container { padding-top: 2rem; max-width: 800px; }
        .app-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; text-align: center; }
        .app-subtitle { color: #8a8f98; margin-top: 0.2rem; margin-bottom: 1.5rem; text-align: center; }
        .status-pill {
            display: inline-block; padding: 4px 12px; border-radius: 999px;
            font-size: 0.8rem; font-weight: 600;
        }
        .status-ready { background-color: #d1fae5; color: #065f46; }
        .status-empty { background-color: #fee2e2; color: #991b1b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm(model_name: str, temperature: float):
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions based only on the "
            "provided context. If the answer is not present in the context, say "
            "'I don't know based on the document provided.'",
        ),
        ("human", "{context}\n\nQuestion: {question}"),
    ]
)


def build_vectorstore(uploaded_file, chunk_size: int, chunk_overlap: int):
    """Save the uploaded PDF, split it, embed it, and build a Chroma index."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        persist_dir = tempfile.mkdtemp(prefix="chroma_")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=persist_dir,
        )
        return vectorstore, persist_dir, len(chunks)


def answer_question(question: str, retriever, model_name: str, temperature: float) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": question})
    llm = get_llm(model_name, temperature)

    # Run the API call with a hard timeout so a stuck/dropped connection
    # fails fast with a clear message instead of hanging the UI forever.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(llm.invoke, final_prompt)
        try:
            response = future.result(timeout=LLM_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            raise RuntimeError(
                f"No response from the model after {LLM_TIMEOUT_SECONDS}s. "
                "This is usually a network/API issue, not a bug in the app — "
                "check your internet connection and GOOGLE_API_KEY, then try again."
            )
    if isinstance(response.content, list):
        return "\n".join(
        block.get("text", "")
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    )

    return str(response.content)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<H1 class="app-title">📚 Chat with your PDF</H1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Upload a book or paper, build the index, then ask it anything.</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Upload & settings — center column
# --------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF book / paper", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    with st.expander("⚙️ Indexing settings", expanded=False):
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 100, step=50)
        top_k = st.slider("Chunks retrieved per question (k)", 1, 10, 3)
with col2:
    with st.expander("🧠 Model settings", expanded=False):
        model_name = st.text_input("Gemini model", value="gemini-3.6-flash")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.1)
        st.caption("Double-check the model name matches one available on your API key.")

build_clicked = st.button(
    "Build knowledge base", type="primary", use_container_width=True,
    disabled=uploaded_file is None,
)

if build_clicked and uploaded_file is not None:
    with st.spinner("Reading PDF, splitting text, and building the index..."):
        vectorstore, persist_dir, n_chunks = build_vectorstore(
            uploaded_file, chunk_size, chunk_overlap
        )
        st.session_state.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "lambda_mult": 0.5, "fetch_k": max(10, top_k * 3)},
        )
        st.session_state.persist_dir = persist_dir
        st.session_state.indexed_filename = uploaded_file.name
        st.session_state.messages = []
    st.success(f"Indexed {n_chunks} chunks from **{uploaded_file.name}**")

status_col, clear_col = st.columns([3, 1])
with status_col:
    if st.session_state.retriever is not None:
        st.markdown(
            f'<span class="status-pill status-ready">● Ready — {st.session_state.indexed_filename}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill status-empty">● No document indexed yet</span>',
            unsafe_allow_html=True,
        )
with clear_col:
    if st.session_state.messages:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

st.divider()

# --------------------------------------------------------------------------
# Chat — center column
# --------------------------------------------------------------------------
if st.session_state.retriever is None:
    st.info("Upload a PDF above and click **Build knowledge base** to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.perf_counter()
                try:
                    answer = answer_question(
                        question, st.session_state.retriever, model_name, temperature
                    )
                except Exception as e:
                    answer = f"⚠️ {e}"
                elapsed = time.perf_counter() - start
            st.markdown(answer)
            st.caption(f"Answered in {elapsed:.1f}s")

        st.session_state.messages.append({"role": "assistant", "content": answer})