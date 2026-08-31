from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from src.config import LLM_MODEL, OPENAI_API_KEY
from src.downstream.tools import retrieve_info, validate_db
from src.log_utils import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful RAG assistant. Your job is to answer user questions "
    "based on documents in the vector database. "
    "Use the 'validate_db' tool first to check if the database is ready. "
    "Then use the 'retrieve_info' tool to find relevant information for the user's query. "
    "If no relevant information is found, politely inform the user that you cannot answer "
    "the question based on the available documents. Do not make up information."
)


def build_downstream_agent():
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
    )

    tools = [validate_db, retrieve_info]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


def run_downstream(query: str):
    agent = build_downstream_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content
