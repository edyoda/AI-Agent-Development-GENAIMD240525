from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


import os
from model_inference import embed_model, llm
from langchain.chains.question_answering import load_qa_chain


# Load and chunk documents
# loader = TextLoader("LangChain_Agent_tools/database/sample_doc.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)


persist_directory = os.path.join(os.path.dirname(__file__), "database", "chroma_langchain_db")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embed_model,
    persist_directory=persist_directory,  # Where to save data locally, remove if not necessary
)


# Add documents to the vector store
# vector_store.add_documents(texts)
#######################################

# question = "what are some of benefits of being a generalist?"
# retrieved_documents = vector_store.similarity_search_with_relevance_scores(
#                         query= question,
#                         k=5,
#                     )

# for doc, score in retrieved_documents:
#     print(f"Document: {doc.page_content[:50]}\nScore: {score}\n")


# relevant_doc = [doc for doc, _ in retrieved_documents]  # Extract documents only


# from langchain.chains.question_answering import load_qa_chain

# chain = load_qa_chain(llm, chain_type="stuff")

# response = chain.run(input_documents = relevant_doc, question=question)

# print(f"Response: {response}\n")



#######################################################

def rag_tool(question: str):
    """
    Function to perform RAG using the vector store.
    
    Args:
        question (str): The query to search for.
        k (int): Number of documents to retrieve.
        
    Returns:
        str: The response generated by the LLM based on the retrieved documents.
    """


    retrieved_documents = vector_store.similarity_search(
        query=question,
        k=5,
    )

    # chain = load_qa_chain(llm, chain_type="stuff")

    # response = chain.run(input_documents = retrieved_documents, question=question)
    
    # return response
    return [doc.page_content for doc in retrieved_documents]


from langchain.agents import Tool

rag_tool_instance = Tool(
    name="RAGTool",
    func=rag_tool,
    description="Useful for performing RAG using a vector store",
)


from langchain.agents import AgentType, initialize_agent

agent = initialize_agent(
    tools=[rag_tool_instance],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# response = agent.run("What are some of the disadvantages of being a generalist?")

# print(f"Agent Response: {response}\n")
