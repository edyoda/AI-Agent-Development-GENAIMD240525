from typing import List, Sequence

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os, asyncio, requests

# Note: This example uses mock tools instead of real APIs for demonstration purposes
def search_web_tool(query: str) -> str:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "API key not found. Set SERPAPI_API_KEY in your environment."

    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": 3,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        snippets = []
        for result in results.get("organic_results", []):
            if "snippet" in result:
                snippets.append(result["snippet"])
        return "\n".join(snippets[:3]) if snippets else "No useful information found."
    except Exception as e:
        return f"Search failed: {e}"


def percentage_change_tool(start: float, end: float) -> float:
    return ((end - start) / start) * 100


model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY'),)

# Using a more creative temperature for the deepseek model
'''
model_client = OpenAIChatCompletionClient(
    model="deepseek-chat",
    temperature=0.85,
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model_info={
        "model_name": "deepseek-chat",
        "api_type": "openai",
        "vendor": "deepseek",
        "json_output": False,
        "vision": False,
        "function_calling": True,
        "function_call": False,
        "streaming": True,
        "cost_per_token": 0,
        "family": "deepseek",
    }
)
'''

planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=model_client,
    system_message="""
    You are a planning agent.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        WebSearchAgent: Searches for information
        DataAnalystAgent: Performs calculations

    You only plan and delegate tasks - you do not execute them yourself.

    When assigning tasks, use this format:
    1. <agent> : <task>

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
)

web_search_agent = AssistantAgent(
    "WebSearchAgent",
    description="An agent for searching information on the web.",
    tools=[search_web_tool],
    model_client=model_client,
    system_message="""
    You are a web search agent.
    Your only tool is search_tool - use it to find information.
    You make only one search call at a time.
    Once you have the results, you never do calculations based on them.
    """,
)

data_analyst_agent = AssistantAgent(
    "DataAnalystAgent",
    description="An agent for performing calculations.",
    model_client=model_client,
    tools=[percentage_change_tool],
    system_message="""
    You are a data analyst.
    Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
    If you have not seen the data, ask for it.
    """,
)


text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination


selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""


team = SelectorGroupChat(
    [planning_agent, web_search_agent, data_analyst_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
)


#task = "Who will become the next Prime Minister of India?"
task = input()

async def main():
    async for message in team.run_stream(task=task):
        print(message)

asyncio.run(main())
