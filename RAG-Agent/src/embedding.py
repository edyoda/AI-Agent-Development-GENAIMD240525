from langchain_openai import OpenAIEmbeddings

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY


def get_embedding_function() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
