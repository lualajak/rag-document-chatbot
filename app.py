from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

import streamlit as st
import tempfile

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


import os

load_dotenv()


st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📑",
    layout="wide"
)

st.title("📑 Chat with your PDF")

@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en-v1.5"
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key = os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens = 256 
    )


def build_retriever(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name


    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()


        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 350
        )

        chunks = splitter.split_documents(
            pages
        )

        embeddings = load_embedding()

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
        )

        retriever = vector_store.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k":5}
        )

        return retriever
    
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.
        Use the provided context to answer with questions about the uploaded file.

        if the user's question is greeting or casual conversation, respond naturally.

        if the answer is not contained in the context, say:
        "i don't know based on the provided document."

        donot makeup answers

        """
     ),
     (
         "human",
         """
        Context:
        {context}

        Question:
        {question} 
        """
     )
])

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

with st.sidebar:
    st.header("Settings")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()



uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:
    if uploaded_file.name !=st.session_state.current_file:
        with st.spinner("Processing PDF...."):
            st.session_state.retriever = build_retriever(
                uploaded_file
            )
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []

        st.success("PDF Ready")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about the document"
)

if question:
    if not st.session_state.retriever:
        st.warning("Please upload a PDF first before asking questions")
    else:
        st.session_state.messages.append(
            {
                "role":"user",
                "content":question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        retriever = st.session_state.retriever

        docs = retriever.invoke(question)

        context = "\n\n".join(doc.page_content for doc in docs)

        llm = load_llm()

        chain = (
            qa_prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content": response
            }
        )