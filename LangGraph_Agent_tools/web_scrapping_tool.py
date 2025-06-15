from langchain_tavily import TavilySearch
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from model_inference import llm
##################################




tool = TavilySearch(
    max_results=2,
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)


agent = initialize_agent(
    tools=[tool],
    llm=llm, 
    agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# agent.invoke("In which year did RCB win the IPL?") 
