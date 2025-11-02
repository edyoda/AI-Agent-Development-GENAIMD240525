## Custom Chatbot to interact with PDF Document


This Chatbot Application enables users to upload PDF Document and get respose for the queries within the provided document. It allows user to upload multiple files at a time and converse to any particular document at a given time. The user also gains the freedom to select from the multiple Large Language Models. 


### Particular of the Application:
- The application is hosted on ```Streamlit``` (for the purpose of showcase the functionality with good enough User Interface)
- ```PyPDF (PdfReader)``` is used for extract information from the PDF document. Also, basic cleaning and processing was done for the content so extracted.
- ```from langchain.text_splitter import RecursiveCharacterTextSplitter``` was used for spliting the extract text into smaller chunks. There can be other methods like sentense splitter or Document specific chunking along with semantic and Agentic Chunking. For the purpose of simplicity, ```RecursiveCharacterTextSplitter``` was used. (Chunk Size: 400 and Chunk Overlap: 40)
- ```Openai Embeddings``` model was used for the purpose of embeddings and these vector were then indexed and stored in ```Pinecone``` vector database.
- User can upload multiple documents at a given time to the Pinecone vector database. Once ready, the chatbot is ready to answer user queries based on the document and the LLM option selected.
- Top 2 document are fetched from the Pinecone vector store for each user query and the response is generated based on user selected Large Language Model. 
- ```load_qa_chain``` from Langchain's chains.question_answering module was used in this for the purpose of simplicity. There can be other way, where I could manually write the prompt as well but for the purpose of simplity I have used load_qa_chain from langchain. **This is modified in the latest commit** 
- ```chat history``` is also maintained in the state of the application. Currently, chathistory is not considered when user ask follow up questions. User can view the chat history right below the response area. 




### Instruction to Use: 
- Provide your OpenAI API key
- Select / Upload the PDF document 
- Click ```'Push for bot to learn'``` on the side bar (wait while the document are processed and vector database is ready)
- Once ready, user can select from the  ```Select document``` and ```Select model``` options. User can now insert question and click ```'Get Answer'```


### Steps to Run Locally:

Run below command in your terminal

- Clone repository

    ```python 
    # clone the github branch
    git clone "repo-link-here"
    ```

- set up environment variables
    ``` python
    PINECONE_API_KEY = "pinecone_api_key_here"
    PINECONE_INDEX_NAME = "pinecone_vector_index_here"
    PINECONE_NAMESPACE = "pinecone_namespace_here"
    ```

- Install dependencies
    ``` python
    pip install -r requirements.txt
    ```

- run command to start the UI
    ``` python
    streamlit run app.py
    ```


---