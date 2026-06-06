from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# ── 1. MODEL ──────────────────────────────────────
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=500
)

# ── 2. PROMPT ─────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are ZyraNova, an elite AI engineering tutor.
Your job:
- Explain concepts clearly with examples
- Give practical coding advice
- Keep responses concise and useful
- Always encourage the student"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# ── 3. PARSER ─────────────────────────────────────
parser = StrOutputParser()

# ── 4. CHAIN ──────────────────────────────────────
chain = prompt | model | parser

# ── 5. MEMORY ─────────────────────────────────────
chat_history = []

# ── 6. CHAT FUNCTION ──────────────────────────────
def chat(user_input):
    response = chain.invoke({
        "input": user_input,
        "history": chat_history
    })
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    return response

# ── 7. MAIN LOOP ───────────────────────────────────
print("=" * 50)
print("        ZyraNovaAI — Your AI Tutor")
print("=" * 50)
print("Type 'quit' to exit | 'clear' to reset memory\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        print("ZyraNova: Goodbye Siddharth! Keep building! 🚀")
        break

    elif user_input.lower() == "clear":
        chat_history.clear()
        print("ZyraNova: Memory cleared! Fresh start.\n")
        continue

    elif user_input == "":
        continue

    response = chat(user_input)
    print(f"\nZyraNova: {response}\n")
    print("-" * 50)