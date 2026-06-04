from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Create LangChain model — connects to Groq
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=500
)

# Simple invoke — send message, get response
response = model.invoke("What is LangChain in one sentence?")

print(response.content)
print(type(response))

from langchain_core.prompts import ChatPromptTemplate

# Create reusable prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI tutor. Explain concepts clearly."),
    ("human", "Explain {topic} for a {level} student in 3 lines.")
])

# Test with 3 different topics — same template reused
topics = [
    {"topic": "neural networks", "level": "beginner"},
    {"topic": "RAG systems", "level": "intermediate"},
    {"topic": "transformers", "level": "advanced"}
]

for t in topics:
    filled = prompt.format_messages(**t)
    response = model.invoke(filled)
    print(f"\n{t['topic'].upper()} ({t['level']}):")
    print(response.content)
    print("-" * 50)

    from langchain_core.output_parsers import StrOutputParser

    # Output parser
    parser = StrOutputParser()

    # New prompt
    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI tutor."),
        ("human", "Explain {topic} in exactly 2 sentences.")
    ])

    # THE CHAIN — prompt → model → parser
    chain = prompt2 | model | parser

    # Run it
    result = chain.invoke({"topic": "vector databases"})

    print("=== CHAIN RESULT ===")
    print(result)
    print(type(result))  # should be plain string now

    from langchain_core.prompts import MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage

    # Prompt with memory slot
    memory_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Aria, a helpful AI tutor."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    memory_chain = memory_prompt | model | parser

    # Memory list
    chat_history = []


    def chat_with_memory(user_input):
        response = memory_chain.invoke({
            "input": user_input,
            "history": chat_history
        })

        # Save both sides to memory
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        return response


    # Test memory across 3 turns
    print("=== MEMORY TEST ===")
    print("Turn 1:", chat_with_memory("My name is Siddharth and I am from Pune"))
    print("---")
    print("Turn 2:", chat_with_memory("What is my name?"))
    print("---")
    print("Turn 3:", chat_with_memory("Where am I from?"))