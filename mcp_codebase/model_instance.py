

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()



# already set `OPENAI_API_KEY`` environment variable

llm = ChatOpenAI(model="gpt-4o-mini")

# print(llm.invoke("hi").content)