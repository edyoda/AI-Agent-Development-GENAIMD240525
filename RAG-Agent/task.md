Your task is to help me build an entire RAG pipeline.

Plan of execution: 
The pipeline should have two parts, which can be triggered independently, upstream and downstream.
- The upstream will process the document including: Parsing, chunking, and adding them to vector database.
- The downstream will trigger with user query and filters, pull relevant chunks from vectordb basis similarity threshold and top_k, to generate the response. 

## upstream breakdown: 
The upstream should be a graph which shall trigger with a document path. once the path is received, it trigger following nodes in order:
- parse_docs: use only pdf for now, and use simple pypdf to parse the document.
- chunk_docs: Use recurssive character text splitter chunking strategy. metadata should be doc_name, 
- add_to_vectordb: use chromadb as vectordb

## downstream breakdown: 
This should be a react agent with two tools: 
- validate_db: inside this it check for the collection if it exist or not. return status if it is ready or not.
- retrieve_info: takes user query to retrieve relevant information and return this back to the agent.

The agent generate final respnse for the user. If in case no relevant information is found, it should deny to answer. 

NOTE: You can use/initialize common parameters like embedding object, collection name, vectordb object etc.

## Default config and Instructions: 
- only coonsider pdf for processing
- chunk size of 200 character with 30 overlap, configurable through env
- collection name should be configurable through env
- use openai ada model for embedding.
- default similarity score threshold can be 0.5 with top k of 3
- before uploading the doc, check if doc with same name exist in the vectordb, if it exist, it should deny the upload.
- add proper logging in the codebase to track progress
- output structure: 


You must use openai text-embedding-3-small model for embeddings and gpt-4.1 model for resposne agent. this should be again configurable from .env file. 