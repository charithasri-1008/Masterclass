import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ---------- CONFIG ----------
CORPUS_PATH = "/kaggle/input/competitions/niat-masterclass-rag-challenge/zyro-dynamics-hr-corpus"

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ---------- LOAD DOCUMENTS ----------
@st.cache_resource
def load_rag():

    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":3}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )

    return retriever, llm

retriever, llm = load_rag()

# ---------- PROMPTS ----------

rag_prompt = ChatPromptTemplate.from_template("""
You are an HR assistant.

Answer ONLY from the given HR policy.

If the answer is not available, reply:

"I don't know."

Context:
{context}

Question:
{question}
""")

guard_prompt = ChatPromptTemplate.from_template("""
Determine whether this question is related to HR policies.

Question:
{question}

Answer only YES or NO.
""")

parser = StrOutputParser()

# ---------- FUNCTIONS ----------

def ask_bot(question):

    decision = (
        guard_prompt
        | llm
        | parser
    ).invoke({"question":question})

    if "NO" in decision.upper():
        return "Sorry, I can answer only HR policy questions."

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    chain = rag_prompt | llm | parser

    return chain.invoke({
        "context":context,
        "question":question
    })

# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="HR Policy Chatbot")

st.title("🤖 HR Policy Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask your HR question...")

if question:

    st.session_state.messages.append(
        {"role":"user","content":question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    answer = ask_bot(question)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )
