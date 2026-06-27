import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# Configuration
# ----------------------------

LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"

CORPUS_PATH = "/kaggle/input/competitions/niat-masterclass-rag-challenge/zyro-dynamics-hr-corpus"

# ----------------------------
# Load API Key
# ----------------------------

try:
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()

    if LLM_PROVIDER == "groq":
        os.environ["GROQ_API_KEY"] = secrets.get_secret("GROQ_API_KEY")

    elif LLM_PROVIDER == "gemini":
        os.environ["GOOGLE_API_KEY"] = secrets.get_secret("GOOGLE_API_KEY")

    elif LLM_PROVIDER == "openai":
        os.environ["OPENAI_API_KEY"] = secrets.get_secret("OPENAI_API_KEY")

except:
    pass

# ----------------------------
# Build Vector Store
# ----------------------------

@st.cache_resource
def load_retriever():

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

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k":4}
    )

retriever = load_retriever()

# ----------------------------
# LLM
# ----------------------------

if LLM_PROVIDER == "groq":

    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=512
    )

elif LLM_PROVIDER == "gemini":

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.1,
        max_output_tokens=512
    )

elif LLM_PROVIDER == "openai":

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=512
    )

# ----------------------------
# Prompts
# ----------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
'''
You are an HR assistant for Zyro Dynamics.

Answer ONLY from the provided context.

If the answer is not found in the context, reply:

"I couldn't find this information in the HR policy documents."

Context:
{context}

Question:
{question}

Answer:
'''
)

OOS_PROMPT = ChatPromptTemplate.from_template(
'''
Determine whether the question is related to Zyro Dynamics HR policies.

Question:
{question}

Reply ONLY with:

YES

or

NO
'''
)

REFUSAL = "I'm sorry, but I can only answer questions related to Zyro Dynamics HR policies."

# ----------------------------
# Functions
# ----------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_chain(question):

    docs = retriever.invoke(question)

    context = format_docs(docs)

    chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "context": context,
            "question": question
        }
    )


def ask_bot(question):

    chain = (
        OOS_PROMPT
        | llm
        | StrOutputParser()
    )

    decision = chain.invoke(
        {
            "question": question
        }
    ).strip().upper()

    if "NO" in decision:
        return REFUSAL

    return rag_chain(question)

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(
    page_title="Zyro HR Assistant",
    page_icon="🤖"
)

st.title("🤖 Zyro Dynamics HR Assistant")

st.write("Ask any question related to Zyro Dynamics HR policies.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask your HR question...")

if prompt:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            response = ask_bot(prompt)

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response
        }
    )