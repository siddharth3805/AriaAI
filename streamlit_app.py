import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

# ── PAGE CONFIG ───────────────────────────────────
st.set_page_config(
    page_title="AriaAI",
    page_icon="🤖",
    layout="centered"
)

# ── HEADER ────────────────────────────────────────
st.title("🤖 AriaAI")
st.caption("Your intelligent AI assistant — chat or upload a PDF")
st.divider()

# ── AI SETUP ──────────────────────────────────────
@st.cache_resource
def load_model():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=500
    )

model = load_model()
parser = StrOutputParser()

# ── PROMPTS ───────────────────────────────────────
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Aria, an elite AI assistant.
Be helpful, concise, and friendly.
If a PDF context is provided, answer from it.
Otherwise answer from your knowledge."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Aria. Answer using ONLY this document context:
{context}
If answer not in context say: 'This is not in the document.'"""),
    ("human", "{question}")
])

chat_chain = chat_prompt | model | parser
rag_chain = rag_prompt | model | parser

# ── SESSION STATE ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_collection" not in st.session_state:
    st.session_state.pdf_collection = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── SIDEBAR ───────────────────────────────────────
with st.sidebar:
    st.header("📄 PDF Mode")
    st.caption("Upload a PDF to ask questions from it")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf"
    )

    if uploaded_file:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                # Save temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Load and chunk
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.split_documents(pages)

                # Store in ChromaDB
                client = chromadb.Client()
                try:
                    client.delete_collection("pdf_upload")
                except:
                    pass
                collection = client.create_collection("pdf_upload")
                collection.add(
                    documents=[c.page_content for c in chunks],
                    ids=[f"c_{i}" for i in range(len(chunks))]
                )

                st.session_state.pdf_collection = collection
                st.session_state.pdf_name = uploaded_file.name
                os.unlink(tmp_path)

            st.success(f"✅ {uploaded_file.name} loaded!")
            st.caption(f"{len(pages)} pages · {len(chunks)} chunks")

    if st.session_state.pdf_name:
        st.info(f"📄 Active: {st.session_state.pdf_name}")
        if st.button("Remove PDF"):
            st.session_state.pdf_collection = None
            st.session_state.pdf_name = None
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.caption("Built with LangChain + Groq + Streamlit")

# ── CHAT HISTORY DISPLAY ──────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── CHAT INPUT ────────────────────────────────────
if prompt := st.chat_input("Ask Aria anything..."):

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # PDF mode or normal mode
            if st.session_state.pdf_collection:
                results = st.session_state.pdf_collection.query(
                    query_texts=[prompt],
                    n_results=3
                )
                context = "\n\n".join(
                    results['documents'][0]
                )
                response = rag_chain.invoke({
                    "question": prompt,
                    "context": context
                })
            else:
                response = chat_chain.invoke({
                    "input": prompt,
                    "history": st.session_state.chat_history
                })

            st.markdown(response)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    st.session_state.chat_history.append(
        HumanMessage(content=prompt)
    )
    st.session_state.chat_history.append(
        AIMessage(content=response)
    )