
"""
Streamlit UI for the PDF RAG (Retrieval-Augmented Generation) app.

Features:
- Upload a PDF
- Build a Chroma vector index
- Ask questions about the PDF
- Gemini-powered answers
- MMR retrieval
- Source citations with page numbers
- View retrieved source snippets
"""

import asyncio
import os
import platform
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


# ---------------------------------------------------------
# Basic setup
# ---------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message="Direct use of automatic function calling"
)

load_dotenv()

# Windows asyncio fix
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )
    except AttributeError:
        pass


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

NOT_FOUND_TEXT = "I don't know based on the document provided."
SNIPPET_CHARS = 350

# Gemini request timeout
LLM_TIMEOUT_SECONDS = 90


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📚",
    layout="centered",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            max-width: 800px;
        }

        .app-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0;
            text-align: center;
        }

        .app-subtitle {
            color: #8a8f98;
            margin-top: 0.2rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .status-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .status-ready {
            background-color: #d1fae5;
            color: #065f46;
        }

        .status-empty {
            background-color: #fee2e2;
            color: #991b1b;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None

if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ---------------------------------------------------------
# Gemini model
# ---------------------------------------------------------

def get_llm(model_name: str, temperature: float):

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY was not found. "
            "Check your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        max_retries=2,
    )


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
You are a helpful assistant that answers questions ONLY
using the provided document context.

The context contains numbered sources such as:
[1], [2], [3].

Each source also contains its page number.

Rules:

1. Answer using only the provided context.
2. If you use information from a source, cite it immediately
   after the relevant sentence.
3. Use citations like [1] or [1][2].
4. Never invent citation numbers.
5. If the answer is not present in the document, say:
   "{NOT_FOUND_TEXT}"
6. Do not use outside knowledge.
""",
        ),
        (
            "human",
            """
Context:

{context}

Question:
{question}
""",
        ),
    ]
)


# ---------------------------------------------------------
# Build vector database
# ---------------------------------------------------------

def build_vectorstore(
    uploaded_file,
    chunk_size: int,
    chunk_overlap: int
):

    with tempfile.TemporaryDirectory() as tmp_dir:

        pdf_path = os.path.join(
            tmp_dir,
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        if not docs:
            raise RuntimeError(
                "The PDF could not be read or contains no text."
            )

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise RuntimeError(
                "No text chunks were created from the PDF."
            )

        # Create temporary Chroma database
        persist_dir = tempfile.mkdtemp(
            prefix="chroma_"
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=persist_dir,
        )

        return (
            vectorstore,
            persist_dir,
            len(chunks)
        )


# ---------------------------------------------------------
# Convert documents to source information
# ---------------------------------------------------------

def docs_to_sources(docs):

    sources = []

    for i, doc in enumerate(docs, start=1):

        page = doc.metadata.get("page")

        page_label = (
            page + 1
            if isinstance(page, int)
            else "?"
        )

        text = " ".join(
            doc.page_content.split()
        )

        snippet = (
            text[:SNIPPET_CHARS]
            + ("..." if len(text) > SNIPPET_CHARS else "")
        )

        sources.append(
            {
                "id": i,
                "page": page_label,
                "snippet": snippet,
            }
        )

    return sources


# ---------------------------------------------------------
# Build context for Gemini
# ---------------------------------------------------------

def build_context(docs):

    parts = []

    for i, doc in enumerate(docs, start=1):

        page = doc.metadata.get("page")

        page_label = (
            page + 1
            if isinstance(page, int)
            else "?"
        )

        parts.append(
            f"[{i}] (Page {page_label})\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------
# Render sources
# ---------------------------------------------------------

def render_sources(
    sources,
    answer
):

    if not sources:
        return

    if NOT_FOUND_TEXT in answer:
        return

    pages = sorted(
        {
            source["page"]
            for source in sources
            if isinstance(source["page"], int)
        }
    )

    if pages:
        st.caption(
            "📄 Pages referenced: "
            + ", ".join(
                str(page)
                for page in pages
            )
        )

    with st.expander(
        f"📚 View Sources ({len(sources)})"
    ):

        for source in sources:

            st.markdown(
                f"**[{source['id']}] Page {source['page']}**"
            )

            st.caption(
                source["snippet"]
            )


# ---------------------------------------------------------
# Ask question
# ---------------------------------------------------------

def answer_question(
    question: str,
    retriever,
    model_name: str,
    temperature: float
):

    # -----------------------------
    # Retrieval
    # -----------------------------

    start_retrieval = time.perf_counter()

    docs = retriever.invoke(question)

    retrieval_time = (
        time.perf_counter()
        - start_retrieval
    )

    if not docs:
        return (
            NOT_FOUND_TEXT,
            [],
            retrieval_time,
            0
        )

    # -----------------------------
    # Sources
    # -----------------------------

    sources = docs_to_sources(docs)

    # -----------------------------
    # Context
    # -----------------------------

    context = build_context(docs)

    # -----------------------------
    # Prompt
    # -----------------------------

    final_prompt = PROMPT.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # -----------------------------
    # Gemini
    # -----------------------------

    start_llm = time.perf_counter()

    try:

        llm = get_llm(
            model_name,
            temperature
        )

        response = llm.invoke(
            final_prompt,
            timeout=LLM_TIMEOUT_SECONDS
        )

    except Exception as e:

        raise RuntimeError(
            f"Gemini API error: {str(e)}"
        ) from e

    llm_time = (
        time.perf_counter()
        - start_llm
    )

    # -----------------------------
    # Extract response text
    # -----------------------------

    if isinstance(response.content, list):

        text = "\n".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    else:

        text = str(
            response.content
        )

    if not text.strip():

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return (
        text,
        sources,
        retrieval_time,
        llm_time
    )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<H1 class="app-title">📚 Chat with your PDF</H1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="app-subtitle">'
    'Upload a book or paper, build the index, then ask it anything.'
    '</p>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF book / paper",
    type=["pdf"]
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    with st.expander(
        "⚙️ Indexing settings",
        expanded=False
    ):

        chunk_size = st.slider(
            "Chunk size",
            300,
            2000,
            1000,
            step=100
        )

        chunk_overlap = st.slider(
            "Chunk overlap",
            0,
            500,
            100,
            step=50
        )

        top_k = st.slider(
            "Chunks retrieved per question (k)",
            1,
            10,
            3
        )


with col2:

    with st.expander(
        "🧠 Model settings",
        expanded=False
    ):

        model_name = st.text_input(
            "Gemini model",
            value="gemini-3.5-flash-lite"
        )

        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.2,
            step=0.1
        )

        st.caption(
            "Use a Gemini model available to your API key."
        )


# ---------------------------------------------------------
# Build button
# ---------------------------------------------------------

build_clicked = st.button(
    "Build knowledge base",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None,
)


# ---------------------------------------------------------
# Build knowledge base
# ---------------------------------------------------------

if build_clicked and uploaded_file is not None:

    try:

        with st.spinner(
            "Reading PDF, splitting text, and building the index..."
        ):

            (
                vectorstore,
                persist_dir,
                n_chunks
            ) = build_vectorstore(
                uploaded_file,
                chunk_size,
                chunk_overlap
            )

            st.session_state.retriever = (
                vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": top_k,
                        "lambda_mult": 0.5,
                        "fetch_k": max(
                            10,
                            top_k * 3
                        ),
                    },
                )
            )

            st.session_state.persist_dir = (
                persist_dir
            )

            st.session_state.indexed_filename = (
                uploaded_file.name
            )

            st.session_state.messages = []

        st.success(
            f"Indexed {n_chunks} chunks from "
            f"**{uploaded_file.name}**"
        )

    except Exception as e:

        st.error(
            f"❌ Indexing failed: {e}"
        )


# ---------------------------------------------------------
# Status
# ---------------------------------------------------------

status_col, clear_col = st.columns(
    [3, 1]
)

with status_col:

    if st.session_state.retriever is not None:

        st.markdown(
            f'''
            <span class="status-pill status-ready">
                ● Ready — {st.session_state.indexed_filename}
            </span>
            ''',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '''
            <span class="status-pill status-empty">
                ● No document indexed yet
            </span>
            ''',
            unsafe_allow_html=True
        )


with clear_col:

    if st.session_state.messages:

        if st.button(
            "🗑️ Clear chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()


# ---------------------------------------------------------
# Divider
# ---------------------------------------------------------

st.divider()


# ---------------------------------------------------------
# Chat
# ---------------------------------------------------------

if st.session_state.retriever is None:

    st.info(
        "Upload a PDF above and click "
        "**Build knowledge base** to get started."
    )

else:

    # Replay previous messages
    for msg in st.session_state.messages:

        with st.chat_message(
            msg["role"]
        ):

            st.markdown(
                msg["content"]
            )

            if msg["role"] == "assistant":

                render_sources(
                    msg.get("sources", []),
                    msg["content"]
                )

    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching document and asking Gemini..."
            ):

                start = time.perf_counter()

                try:

                    (
                        answer,
                        sources,
                        retrieval_time,
                        llm_time
                    ) = answer_question(
                        question,
                        st.session_state.retriever,
                        model_name,
                        temperature
                    )

                except Exception as e:

                    answer = (
                        f"⚠️ {e}"
                    )

                    sources = []

                    retrieval_time = 0

                    llm_time = 0

                elapsed = (
                    time.perf_counter()
                    - start
                )

            st.markdown(answer)

            render_sources(
                sources,
                answer
            )

            if llm_time > 0:

                st.caption(
                    f"⏱️ Total: {elapsed:.1f}s | "
                    f"Retrieval: {retrieval_time:.1f}s | "
                    f"Gemini: {llm_time:.1f}s"
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

