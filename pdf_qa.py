from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

# ── 1. SETUP ──────────────────────────────────────
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=500
)
parser = StrOutputParser()


# ── 2. LOAD PDF ───────────────────────────────────
def load_pdf(pdf_path):
    print(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"✅ Loaded {len(pages)} pages")
    return pages


# ── 3. CHUNK THE PDF ──────────────────────────────
def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # each chunk = 500 characters
        chunk_overlap=50,  # 50 chars overlap between chunks
        length_function=len
    )
    chunks = splitter.split_documents(pages)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ── 4. STORE IN CHROMADB ──────────────────────────
def store_in_chromadb(chunks, collection_name="pdf_knowledge"):
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection if exists
    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Add chunks to ChromaDB
    documents = [chunk.page_content for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(documents=documents, ids=ids)
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB\n")

    return collection


# ── 5. RAG PROMPT ─────────────────────────────────
pdf_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Aria, an AI assistant that answers questions from documents.
Use ONLY the provided context to answer.
If the answer is not in the context, say "This information is not in the document."
Always be accurate and cite relevant details from the context.

Context from document:
{context}"""),
    ("human", "{question}")
])

pdf_chain = pdf_prompt | model | parser


# ── 6. ANSWER QUESTION ────────────────────────────
def answer_question(collection, question):
    # Find relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    # Build context
    context = "\n\n".join(results['documents'][0])

    # Generate answer
    answer = pdf_chain.invoke({
        "question": question,
        "context": context
    })

    return answer


# ── 7. MAIN PDF QA SYSTEM ─────────────────────────
def run_pdf_qa(pdf_path):
    print("=" * 50)
    print("     AriaAI — PDF Question Answering")
    print("=" * 50 + "\n")

    # Load and process PDF
    pages = load_pdf(pdf_path)
    chunks = chunk_documents(pages)
    collection = store_in_chromadb(chunks)

    print("Ready! Ask questions about your PDF.")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if question == "":
            continue

        answer = answer_question(collection, question)
        print(f"\nAria: {answer}\n")
        print("-" * 50)


# ── 8. RUN IT ─────────────────────────────────────
# Change this to your PDF path
PDF_PATH = "sample.pdf"  # add any PDF later

if os.path.exists(PDF_PATH):
    run_pdf_qa(PDF_PATH)
else:
    print("❌ No PDF found!")
    print("👉 Add a PDF file to your project folder")
    print(f"👉 Name it 'sample.pdf'")
    print("👉 Then run this file again")