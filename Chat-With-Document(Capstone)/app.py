import streamlit as st
import os
from utils import *
import uuid

st.set_page_config(layout="wide")

def main():
    st.title("Custom Chatbot")

    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state["OPENAI_API_KEY"]=''
    if "unique_id" not in st.session_state:
        st.session_state["unique_id"]=''
    if "bot_ready_flag" not in st.session_state:
        st.session_state["bot_ready_flag"]=''
    if "chathistory" not in st.session_state:
        st.session_state["chathistory"] = {}

    st.sidebar.title("😎🗝️")

    st.session_state["OPENAI_API_KEY"]=st.sidebar.text_input("Enter your OPENAI API key", type="password")
    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]
    

    st.session_state["unique_id"] = uuid.uuid4().hex
    unique_id = st.session_state["unique_id"]
    os.environ["NAMESPACE"] = unique_id

    files = st.sidebar.file_uploader("Upload your file", type="pdf",accept_multiple_files=True)

    push_doc = st.sidebar.button("Push for bot to learn")
    if push_doc and files:
        st.session_state["bot_ready_flag"] = False
        # create doc out of provided pdf
        docs = create_docs(files,unique_id)
        global tiny_docs
        tiny_docs = split_docs(docs, chunk_size=400, chunk_overlap=40)

    #     # # creating embeddings
        embedding = get_embeddings()

        with st.spinner("Wait! ChatBot is Learning ✋🏻"):
            push_status = push_to_pinecone(tiny_docs, embedding, os.environ.get("NAMESPACE"))
            if push_status:
                st.session_state["bot_ready_flag"] = True
            else:
                st.warning("Something is not right with OpenAI API Key Provided")

    submit = None
    st.subheader("Search Document..🤖")
    if files:
        col1, col2 = st.columns(2)
        with col1:
            files_name = [file.name for file in files]
            selected_file_name = st.selectbox("Select Document to interact with..", 
                            files_name,
                            index=None,
                            placeholder="Click here to select document")
            st.write(f"*selected file: {selected_file_name}*")
        with col2:
            models_list = ["gpt-3.5-turbo-0125", "gpt-4o-mini"]
            selected_model = st.selectbox("Select the model you wish to choose",
                                          models_list,
                                          index=None,
                                          placeholder="Click here to select the model")
            st.markdown("*NOTE: Model is set to default temperature of 0.5*")
    

        st.markdown("---")

        global query
        query = st.text_area("Enter query here.. 👇🏻", key='question')

        submit = st.button("Get Answer")

    if submit and st.session_state["bot_ready_flag"]==True:
        if query:
            # text embedding
            embedding = get_embeddings()

            # get similar docs
            with st.spinner("Generating Response"):
                response, relevant_docs = get_answer(selected_model, query, embedding, selected_file_name, k=2)

                st.write(response)
                if selected_file_name not in st.session_state.chathistory:
                    st.session_state.chathistory[selected_file_name] = [{query:response}]
                else:
                    st.session_state.chathistory[selected_file_name].append({query:response})

                # with st.expander(label="See relevant documents", expanded=False):
                #     # st.write(relevant_docs)
                #     for doc in relevant_docs:
                #         st.write(doc.page_content)
                #         st.markdown("---")

                st.markdown("---")
                if selected_file_name in st.session_state.chathistory:
                    st.markdown("### ChatHistory")
                    for i, msg in enumerate(st.session_state.chathistory[selected_file_name]):
                        for question, response in msg.items():
                            st.chat_message("user").write(question)
                            st.chat_message("Assistant").write(response)
            
            try:
                st.session_state["question"] = ""
            except:
                pass
        else:
            st.error("You gotta be kidding me.. I really wish I could read your mind")
    elif submit:
        st.error("Push the document first")
    


if __name__ == "__main__":
    main()
