import streamlit as st
from groq import Groq
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
    page_title="ZyraNovaAI",
    page_icon="🤖",
    layout="centered"
)

# ── HEADER ────────────────────────────────────────
st.title("🤖 ZyraNovaAI")
st.caption("Your intelligent AI assistant — chat or upload a PDF")
st.divider()

# ── AI SETUP ──────────────────────────────────────
@st.cache_resource
def load_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key)

client = load_client()

# ── SESSION STATE ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_collection" not in st.session_state:
    st.session_state.pdf_collection = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── CHAT FUNCTION ─────────────────────────────────
def get_response(user_input, context=None):
    if context:
        system = f"""You are Zyra, an intelligent AI assistant.
Answer using ONLY this document context:
{context}
If answer not in context say: 'This is not in the document.'"""
    else:
        system = """You are Zyra, an elite AI assistant.
Be helpful, concise, and friendly."""

    # Build messages
    messages = [{"role": "system", "content": system}]

    # Add history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        else:
            messages.append({"role": "assistant", "content": msg.content})

    # Add current message
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content

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
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.split_documents(pages)

                chroma_client = chromadb.Client()
                try:
                    chroma_client.delete_collection("pdf_upload")
                except:
                    pass
                collection = chroma_client.create_collection("pdf_upload")
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

    st.caption("Built with Groq + Streamlit | ZyraNovaAI")

# ── CHAT HISTORY DISPLAY ──────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── CHAT INPUT ────────────────────────────────────
if prompt := st.chat_input("Ask Zyra anything..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.pdf_collection:
                results = st.session_state.pdf_collection.query(
                    query_texts=[prompt],
                    n_results=3
                )
                context = "\n\n".join(results['documents'][0])
                response = get_response(prompt, context=context)
            else:
                response = get_response(prompt)

            st.markdown(response)

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