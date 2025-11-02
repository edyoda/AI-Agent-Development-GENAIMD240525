import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone as PineconeClient
from langchain.chains.question_answering import load_qa_chain
from datetime import datetime
from langchain_community.vectorstores import Pinecone
import os
import time
from dotenv import load_dotenv

load_dotenv()



def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text.replace("\n","")

def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)

        docs.append(Document(
            page_content = chunks,
            metadata = {"name":filename.name, "type=": filename.type, "size": filename.size, "unique_id": unique_id, 'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ))

        
    return docs

# transform documents
def split_docs(documents, chunk_size=400, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    
    return docs

def get_embeddings():
    embedding = OpenAIEmbeddings()
    return embedding


def push_to_pinecone(docs, embedding,namespace):

    pc = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name=os.environ.get("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)

    index_dict = index.describe_index_stats()
    namespace_list = list(index_dict["namespaces"].keys())
    if os.environ.get("PINECONE_NAMESPACE", "edyoda") in namespace_list:
        index.delete(delete_all=True, namespace=os.environ.get("PINECONE_NAMESPACE", "edyoda"))
    else:
        pass
    
    vector = []
    try:
        for i, doc in enumerate(docs):
            entry = { "id": str(i),
                    "values": embedding.embed_query(doc.page_content),
                    "metadata":doc.metadata}
            vector.append(entry)

        
        index = Pinecone.from_documents(docs, embedding, index_name = index_name, namespace=os.environ.get("PINECONE_NAMESPACE", "edyoda"))

        st.sidebar.write("This 10 seconds delay was added Manually... \n(because I'm using some free resources)")
        time.sleep(10)

        return True
    except Exception as e:
        return None



#Function to pull index data from Pinecone
def pull_from_pinecone(embeddings):

    pinecone_apikey = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name =os.environ.get("PINECONE_INDEX_NAME")

    PineconeClient(
    api_key=pinecone_apikey
    )

    #PineconeStore is an alias name of Pinecone class, please look at the imports section at the top :)
    index = Pinecone.from_existing_index(pinecone_index_name, embeddings, namespace=os.environ.get("PINECONE_NAMESPACE", "edyoda"))

    return index




def get_similar_doc(query, embedding, file_name, k=2):

    pc = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name=os.environ.get("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)

    index = pull_from_pinecone(embeddings=embedding)
    similar_doc = index.similarity_search_with_score(query, int(k), filter={"name":file_name})
    
    return [doc for doc, similarity_score in similar_doc]
    

from prompt import CHAT_PROMPT

def get_answer(model_name, query, embedding, file_name, k=2):
    llm=ChatOpenAI(model= model_name ,temperature=0.5)
    # chain = load_qa_chain(llm, chain_type="stuff")
    chain = CHAT_PROMPT | llm

    relevent_doc = get_similar_doc(query, embedding, file_name, k=2)
    response = chain.invoke(input = {"context": relevent_doc, "question":query}, config={"callbacks": []})
    return response.content, relevent_doc




# ----------------------------------------------------------------- #
# ------------------ END OF UTILITY FUNCTIONS --------------------- #
# ----------------------------------------------------------------- #


