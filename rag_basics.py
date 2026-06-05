from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
# This model converts text → numbers
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for sentences
sentences = [
    "I love machine learning",
    "I enjoy deep learning",
    "The weather is nice today",
    "Neural networks are fascinating",
    "It is sunny outside"
]

embeddings = embedder.encode(sentences)

print(f"Each sentence becomes {len(embeddings[0])} numbers")
print(f"\nFirst sentence embedding (first 5 numbers):")
print(embeddings[0][:5])

# Calculate similarity between sentences
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare similarities
print("\n=== SIMILARITY SCORES ===")
base = embeddings[0]  # "I love machine learning"

for i, sentence in enumerate(sentences):
    score = cosine_similarity(base, embeddings[i])
    print(f"{score:.3f} — {sentence}")


import chromadb

# Create ChromaDB client
# This creates a local database folder called 'chroma_db'
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection — like a table in regular database
collection = client.get_or_create_collection(
    name="aria_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# Add documents to the database
documents = [
    "LangChain is a framework for building AI applications",
    "RAG stands for Retrieval Augmented Generation",
    "ChromaDB is an open source vector database",
    "Embeddings convert text into numerical vectors",
    "Python is the most popular language for AI development",
    "LLMs are Large Language Models trained on massive datasets"
]

# Add to ChromaDB
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"Added {len(documents)} documents to ChromaDB")

# Query the database
query = "What is RAG?"

results = collection.query(
    query_texts=[query],
    n_results=2  # return top 2 most similar
)

print(f"\nQuery: {query}")
print("\nMost relevant documents found:")
for doc in results['documents'][0]:
    print(f"→ {doc}")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# ── 2. VECTOR DATABASE ────────────────────────────
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="aria_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# ── 3. KNOWLEDGE BASE ─────────────────────────────
knowledge = [
    "AriaAI is an intelligent AI assistant built with Python and Groq",
    "LangChain is a framework for building production AI applications",
    "RAG stands for Retrieval Augmented Generation — it lets AI answer from documents",
    "ChromaDB is a vector database that stores text as numerical embeddings",
    "Groq API provides fast LLM inference using LLaMA 3 models",
    "Prompt engineering is the art of writing effective instructions for AI",
    "Embeddings are numerical representations of text that capture meaning",
    "Vector similarity search finds documents with similar meaning to a query",
    "LLMs are Large Language Models trained on massive text datasets",
    "Python is the primary language for AI and ML development"
]

# Add knowledge to ChromaDB
collection.add(
    documents=knowledge,
    ids=[f"fact_{i}" for i in range(len(knowledge))]
)
print(f"✅ Knowledge base loaded: {len(knowledge)} facts\n")

# ── 4. RAG PROMPT ─────────────────────────────────
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Aria, an AI tutor.
Answer the question using ONLY the context provided.
If the answer is not in the context, say "I don't have that information."
Always be concise and clear.

Context:
{context}"""),
    ("human", "{question}")
])

rag_chain = rag_prompt | model | parser


# ── 5. RAG FUNCTION ───────────────────────────────
def rag_answer(question):
    # Step 1 — Find relevant documents
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    # Step 2 — Build context from results
    relevant_docs = results['documents'][0]
    context = "\n".join(relevant_docs)

    print(f"📚 Retrieved context:")
    for doc in relevant_docs:
        print(f"  → {doc}")
    print()

    # Step 3 — Generate answer using context
    answer = rag_chain.invoke({
        "question": question,
        "context": context
    })

    return answer


# ── 6. TEST IT ────────────────────────────────────
questions = [
    "What is RAG?",
    "What is ChromaDB?",
    "What is AriaAI?",
    "What is the capital of France?"  # not in knowledge base
]

for q in questions:
    print(f"❓ Question: {q}")
    print(f"💡 Answer: {rag_answer(q)}")
    print("=" * 50 + "\n")