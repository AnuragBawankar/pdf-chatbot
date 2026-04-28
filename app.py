import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except:
    pass

st.set_page_config(page_title="Chat with your Documents")
st.header("Document Chatbot")
st.write("Upload a PDF, Word, or TXT file and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "docx", "txt"]
)


def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "".join([page.extract_text() for page in reader.pages])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")


if uploaded_file is not None:
    text = extract_text(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    st.success(f"'{uploaded_file.name}' processed! Ask me anything about it.")

    question = st.text_input("Ask a question about your document:")

    if question:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        response = llm.invoke(
            f"Answer this question based on the context:\n\nContext: {context}\n\nQuestion: {question}")
        st.write("**Answer:**", response.content)
