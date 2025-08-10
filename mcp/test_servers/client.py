# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent


from model_instance import llm

import asyncio

# async def main(question):
#     server_params = StdioServerParameters(
#         command="python",
#         args=["test_servers/math_server.py"],
#     )

#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             # Initialize the connection
#             await session.initialize()

#             # Get tools
#             tools = await load_mcp_tools(session)

#             # Create and run the agent
#             agent = create_react_agent(llm, tools) # , state_modifier = chat_prompt)
#             agent_response = await agent.ainvoke({"messages": question})
#             return agent_response

# Run the async main function

# question = "what is (3.3 * 2) + 1"
# response = asyncio.run(main(question))
# print(response["messages"][-1].content)







from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent


async def main(question):
    client =  MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["test_servers/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)
    # math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    # weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})

    response = await agent.ainvoke({"messages": question})

    return response

# question = "what is (3.3 * 2) + 1. also what is the weather in bangalore ?"
# response = asyncio.run(main(question))
# print("-*-"*60)
# print(f"Question: {question}\nResponse: ")
# print(response["messages"][-1].content)




# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.graph import StateGraph, MessagesState, START
# from langgraph.prebuilt import ToolNode, tools_condition

# from langchain.chat_models import init_chat_model
# # model = init_chat_model(llm)

# async def main(question):
#     client = MultiServerMCPClient(
#         {
#             "math": {
#                 "command": "python",
#                 # Make sure to update to the full absolute path to your math_server.py file
#                 "args": ["test_servers/math_server.py"],
#                 "transport": "stdio",
#             },
#             "weather": {
#                 # make sure you start your weather server on port 8000
#                 "url": "http://localhost:8000/sse",
#                 "transport": "sse",
#             },

# #             "sqlite": {
# #                 # make sure you start your weather server on port 8000
# #                 "url": "http://localhost:8000/sse",
# #                 "transport": "sse",
#                 # },
#         }
#     )
#     tools = await client.get_tools()

#     def call_model(state: MessagesState):
#         response = llm.bind_tools(tools).invoke(state["messages"])
#         return {"messages": response}

#     builder = StateGraph(MessagesState)
#     builder.add_node(call_model)
#     builder.add_node(ToolNode(tools))
#     builder.add_edge(START, "call_model")
#     builder.add_conditional_edges(
#         "call_model",
#         tools_condition,
#     )
#     builder.add_edge("tools", "call_model")
#     graph = builder.compile()
#     # agent = create_react_agent(llm, tools, state_modifier = chat_prompt)
#     response = await graph.ainvoke({"messages": question})
#     return response
    





# question = "what is (3.3 * 2) + 1. also what is the weather in bangalore ?"
# response = main(question)
# print(response["messages"][-1].content)