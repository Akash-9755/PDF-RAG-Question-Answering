from langchain_community.document_loaders import TextLoader

data = TextLoader("documents/notes.txt")
final = data.load()

print(final[0].page_content)