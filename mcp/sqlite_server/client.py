# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent


from model_instance import llm

import asyncio

from langchain.prompts import ChatPromptTemplate


sys_msg = """You are a Business Intelligence (BI) assistant with access to a SQLite database. Your role is to help users explore the data, extract useful insights, and answer their questions using SQL and available tools.
 
Tools:
- list_tables(statement): Get a list of all tables.
- describe_table(statement): Get schema for a given table.
- read_query(statement): Execute SELECT queries.
- append_insight(statement): Log key business insight.
 
Instructions:
1. Always break down the user request logically before acting.
2. Use list_tables first to see what's available.
3. Use describe_table to explore table structure before querying.
4. Use read_query for SELECT queries.
5. Use append_insight to summarize findings in plain business terms.
6. Handle errors gracefully:
   - Explain failures.
   - Suggest corrections or alternatives.
7. Think step-by-step. Communicate clearly and concisely.
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ('system', sys_msg),
    ("human", "{messages}")
])

async def main(question):
    server_params = StdioServerParameters(
        command="python",
        args= ["sqlite_server/sqlite_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(llm, tools, state_modifier = chat_prompt)
            agent_response = await agent.ainvoke({"messages": question})
            return agent_response

# Run the async main function

# question = "what is the total number of patients?"
# question = "show me what type of encounters are most common among patients with different races"
# question = ""
# response = asyncio.run(main(question))
# print(response["messages"][-1].content)







# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent


# async def main(question):
#     async with MultiServerMCPClient(
#         {
#             "math": {
#                 "command": "python",
#                 # Make sure to update to the full absolute path to your math_server.py file
#                 "args": ["math_server.py"],
#                 "transport": "stdio",
#             },
#             "weather": {
#                 # make sure you start your weather server on port 8000
#                 "url": "http://localhost:8000/sse",
#                 "transport": "sse",
#             }
#         }
#     ) as client:
#         agent = create_react_agent(llm, client.get_tools())
#         # math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
#         # weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})

#         response = await agent.ainvoke({"messages": question})

#         return response, client




# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.graph import StateGraph, MessagesState, START
# from langgraph.prebuilt import ToolNode, tools_condition

# from langchain.chat_models import init_chat_model
# # model = init_chat_model(llm)

# async def main(question):
#     async with MultiServerMCPClient(
#         {
# #             # "math": {
# #             #     "command": "python",
# #             #     # Make sure to update to the full absolute path to your math_server.py file
# #             #     "args": ["math_server.py"],
# #             #     "transport": "stdio",
# #             # },
# #             # "weather": {
# #             #     # make sure you start your weather server on port 8000
# #             #     "url": "http://localhost:8000/sse",
# #             #     "transport": "sse",
# #             # },
#             "sqlite": {
#             "command": "python",
#             "args": ["sqlite_server.py"],
#             "transport": "stdio",
#             },

# #             "sqlite": {
# #                 # make sure you start your weather server on port 8000
# #                 "url": "http://localhost:8000/sse",
# #                 "transport": "sse",
#                 # },
#         }
#     ) as client:
#         tools = client.get_tools()
#         def call_model(state: MessagesState):
#             response = llm.bind_tools(tools).invoke(state["messages"])
#             return {"messages": response}

#         builder = StateGraph(MessagesState)
#         builder.add_node(call_model)
#         builder.add_node(ToolNode(tools))
#         builder.add_edge(START, "call_model")
#         builder.add_conditional_edges(
#             "call_model",
#             tools_condition,
#         )
#         builder.add_edge("tools", "call_model")
#         graph = builder.compile()
#         # agent = create_react_agent(llm, tools, state_modifier = chat_prompt)
#         response = await graph.ainvoke({"messages": question})
#         return response
    





# question = "what is (3.3 * 2) + 1. also what is the weather in bangalore ?"
# response = main(question)
# print(response["messages"][-1].content)