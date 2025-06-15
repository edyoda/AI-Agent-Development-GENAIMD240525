import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

llm = ChatOpenAI(model= 'gpt-4o-mini', temperature=0)
embed_model = OpenAIEmbeddings(model='text-embedding-ada-002')

# response = llm.invoke("hi")

# print(response.content)

# print(len(embed_model.embed_query("What is the capital of India?")))