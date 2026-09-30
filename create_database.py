from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


data = PyPDFLoader("documents/deep-learning-material.pdf")
doc = data.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)


chunks = splitter.split_documents(doc)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="ChromaDB"
)





