from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

vectordb = Chroma(persist_directory='./md_vectors', embedding_function=OpenAIEmbeddings())

query = "How to summarize the average of a field"

docs = vectordb.similarity_search(query, k=5)

print(docs[0].page_content[0:500])
