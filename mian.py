from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

warnings.filterwarnings(
    "ignore",
    message="Direct use of automatic function calling"
)

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2" 
)

vectorstore = Chroma(
    persist_directory="ChromaDB",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.5,
        "fetch_k": 10
        
    }
    
)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# prompt templet

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",""" You are a helpful assistant that answers questions based on the provided context. if ans 
         is not present in the context, say 'I don't know'"""),
        ("human", "{context}\n\nQuestion: {question}")
    ]
)

print("\n===== RAG System is Created =====\n")

print("\n===== RAG System is Ready to Answer Questions =====\n")

print("press 0 to exit the program")

while True:
    query = input("\nEnter your question: ")
    if query == "0":
        break
    
    docs = retriever.invoke(query)
    
    context = "".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" : context,
        "question" : query
    })
    
    response = llm.invoke(final_prompt)
    
    print(f"\n AI: {response.content[0]["text"]}")
