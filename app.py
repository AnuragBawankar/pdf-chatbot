import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

load_dotenv()

st.set_page_config(page_title="Chat with your PDF")
st.header("PDF Chatbot")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(chunks, embeddings)
    st.success("PDF processed! Ask me anything about it.")

    question = st.text_input("Ask a question about your PDF:")

    if question:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        response = llm.invoke(
            f"Answer this question based on the context:\n\nContext: {context}\n\nQuestion: {question}")
        st.write("**Answer:**", response.content)
