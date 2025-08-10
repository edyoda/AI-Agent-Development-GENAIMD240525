import streamlit as st
import asyncio



from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, FunctionMessage
from pprint import pprint


def format_response(response):
    """
    Format the response from the server for better readability.
    """

    formatted_response = ""
    
    for i, message in enumerate(response["messages"]):
        formatted_response += "\n" + f"\nMessage {i + 1} - Type: {type(message).__name__}"
        formatted_response += "\n" + "-" * 50

        if isinstance(message, HumanMessage):
            formatted_response += "\n" + f"User Input: {message.content}"

        elif isinstance(message, AIMessage):
            formatted_response += "\n" + f"AI Response: {message.content}"
            if message.tool_calls:
                formatted_response += "\n" + "Tool Calls:"
                for tool_call in message.tool_calls:
                    formatted_response += "\n" + f"  Tool Name: {tool_call['name']}"
                    formatted_response += "\n" + "  Arguments:"
                    formatted_response += "\n" + str(tool_call['args'])

        elif isinstance(message, ToolMessage):
            formatted_response += "\n" + f"Tool Response from {message.tool_call_id}:"
            formatted_response += "\n" + f"  Content: {message.content}"

        elif isinstance(message, FunctionMessage):
            formatted_response += "\n" + f"Function Response:"
            formatted_response += "\n" + f"  Name: {message.name}"
            formatted_response += "\n" + f"  Content: {message.content}"

        else:
            formatted_response += "\n" + "Other Message:"
            formatted_response += "\n" + message

        formatted_response += "\n" + "-*-" * 60

    return formatted_response


st.header("Model Context Protocol Demo!")
st.markdown("---")


test_content = st.selectbox(
    "Select the type of server you want to use:",
    ("test", "sqlite"),
    index=0,
)
if test_content == "test":
    # test server
    from test_servers.client import main
elif test_content == 'sqlite':
    # sqlite server
    from sqlite_server.client import main


question = st.text_input("Enter your question here:", placeholder="Question")


if st.button("Submit"):
    if question:
        # run main function in async
        response = asyncio.run(main(question))
        st.markdown("---")
        st.subheader("Response:")
        st.write(response["messages"][-1].content)
        with st.expander("Full Response", expanded=False):
            formatted_response = format_response(response)
            st.code(formatted_response, language="python")
    else:
        st.warning("Please enter a question before submitting.")
st.markdown("---")