from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader('./md_docs', glob='**/*.mdx', loader_cls=UnstructuredMarkdownLoader, show_progress=True, loader_kwargs={'mode': 'elements'})

docs = loader.load()
