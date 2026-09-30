from langchain_community.document_loaders import WebBaseLoader

url = "https://www.ibm.com/think/topics/data-science"

loader = WebBaseLoader(url)

doc = loader.load()

print(len(doc))