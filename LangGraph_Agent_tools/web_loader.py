# from langchain.document_loaders import WebBaseLoader
# from langchain.indexes import VectorstoreIndexCreator

from langchain_community.document_loaders import WebBaseLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator

from model_inference import embed_model, llm

loader = WebBaseLoader("https://en.wikipedia.org/wiki/India")


index = VectorstoreIndexCreator(embedding=embed_model).from_loaders([loader])

# query = "what are some of the major rivers in India?"


# response = index.query(query, llm=llm)

# print(response)

