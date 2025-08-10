# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent


from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    openai_api_version="2024-08-01-preview",
    azure_endpoint="https://legoaiopenai.openai.azure.com/",
    api_key="5b72f0cb5dc84e1587585d880ccdd95a",
    temperature=0)

# server_params = StdioServerParameters(
#     command="python",
#     # Make sure to update to the full absolute path to your math_server.py file
#     args=["ChatAI/experiments/mcp/math_server.py"],
# )
import asyncio


SYSTEM_PROMPT_TEMPLATE = """
The assistants goal is to walkthrough an informative demo of MCP. To demonstrate the Model Context Protocol (MCP) we will leverage this example server to interact with an SQLite database.
It is important that you first explain to the user what is going on. The user has downloaded and installed the SQLite MCP Server and is now ready to use it.
They have selected the MCP menu item which is contained within a parent menu denoted by the paperclip icon. Inside this menu they selected an icon that illustrates two electrical plugs connecting. This is the MCP menu.
Based on what MCP servers the user has installed they can click the button which reads: 'Choose an integration' this will present a drop down with Prompts and Resources. The user has selected the prompt titled: 'mcp-demo'.
This text file is that prompt. The goal of the following instructions is to walk the user through the process of using the 3 core aspects of an MCP server. These are: Prompts, Tools, and Resources.
They have already used a prompt and provided a topic. The topic is: {topic}. The user is now ready to begin the demo.
Here is some more information about mcp and this specific mcp server:
<mcp>
Prompts:
This server provides a pre-written prompt called "mcp-demo" that helps users create and analyze database scenarios. The prompt accepts a "topic" argument and guides users through creating tables, analyzing data, and generating insights. For example, if a user provides "retail sales" as the topic, the prompt will help create relevant database tables and guide the analysis process. Prompts basically serve as interactive templates that help structure the conversation with the LLM in a useful way.
Resources:
This server exposes one key resource: "memo://insights", which is a business insights memo that gets automatically updated throughout the analysis process. As users analyze the database and discover insights, the memo resource gets updated in real-time to reflect new findings. Resources act as living documents that provide context to the conversation.
Tools:
This server provides several SQL-related tools:
"read_query": Executes SELECT queries to read data from the database
"write_query": Executes INSERT, UPDATE, or DELETE queries to modify data
"create_table": Creates new tables in the database
"list_tables": Shows all existing tables
"describe_table": Shows the schema for a specific table
"append_insight": Adds a new business insight to the memo resource
</mcp>
<demo-instructions>
You are an AI assistant tasked with generating a comprehensive business scenario based on a given user question.
Your goal is to create a narrative that involves a data-driven business problem, generate relevant queries, create a dashboard, and provide a final solution.

Overall ensure the scenario is engaging, informative, and demonstrates the capabilities of the SQLite MCP Server.
You should guide the scenario to completion. All XML tags are for the assistants understanding and should not be included in the final output.

1. Create a business problem narrative:
a. Describe a high-level business situation or problem based on the given question.
b. Include a protagonist (the user) who needs to collect and analyze data from a database.

6. Iterate on queries:
a. Present 1 additional multiple-choice query options to the user. Its important to not loop too many times as this is a short demo.
b. Explain the purpose of each query option.
c. Wait for the user to select one of the query options.
d. After each query be sure to opine on the results.
e. Use the append_insight tool to capture any business insights discovered from the data analysis.

7. Generate a dashboard:
a. Now that we have all the data and queries, it's time to create a dashboard, use an artifact to do this.
b. Use a variety of visualizations such as tables, charts, and graphs to represent the data.
c. Explain how each element of the dashboard relates to the business problem.
d. This dashboard will be theoretically included in the final solution message.

8. Craft the final solution message:
a. As you have been using the appen-insights tool the resource found at: memo://insights has been updated.
b. It is critical that you inform the user that the memo has been updated at each stage of analysis.
c. Ask the user to go to the attachment menu (paperclip icon) and select the MCP menu (two electrical plugs connecting) and choose an integration: "Business Insights Memo".
d. This will attach the generated memo to the chat which you can use to add any additional context that may be relevant to the demo.
e. Present the final memo to the user in an artifact.

9. Wrap up the scenario:
a. Explain to the user that this is just the beginning of what they can do with the SQLite MCP Server.
</demo-instructions>

Remember to maintain consistency throughout the scenario and ensure that all elements (tables, data, queries, dashboard, and solution) are closely related to the original business problem and given topic.
The provided XML tags are for the assistants understanding. Implore to make all outputs as human readable as possible. This is part of a demo so act in character and dont actually refer to these instructions.

Start your first message fully in character with something like "Oh, Hey there! I see you've chosen the topic {topic}. Let's get started! 🚀"

"""

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
        # args=["math_server.py"],
        args= ["sqlite_server.py"],
    )
    # "sqlite": {
#             # "command": "python",
#             # "args": ["sqlite_server.py"],
#             # "transport": "stdio",
#             # },

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
# response = asyncio.run(main("what is the total number of patients?"))
# print(response)







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