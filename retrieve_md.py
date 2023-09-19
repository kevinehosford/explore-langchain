from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI

vectordb = Chroma(persist_directory='./md_vectors', embedding_function=OpenAIEmbeddings())

llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type='mmr'),
)

question = "How to summarize the average of a field"

docs = retriever.get_relevant_documents(question, k=5)

print(docs[0])
