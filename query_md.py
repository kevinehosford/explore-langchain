from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

vectordb = Chroma(persist_directory='./md_vectors', embedding_function=OpenAIEmbeddings())

query = "summarize"

docs = vectordb.similarity_search(query, k=5)
