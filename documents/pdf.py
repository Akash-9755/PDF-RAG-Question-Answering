from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("documents/GRU.pdf")

doc = data.load()

print(len(doc))