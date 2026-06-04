from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Conversation memory
conversation_history = [
    {
        "role": "system",
        "content": "You are Aria, an expert AI tutor. Explain everything clearly with examples. Keep responses short and practical."
    }
]

def chat(user_message):
    # Add user message to memory
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Send entire history to AI
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=conversation_history,
        max_tokens=500,
        temperature=0.7
    )

    # Extract response
    ai_reply = response.choices[0].message.content

    # Add AI reply to memory too
    conversation_history.append({
        "role": "assistant",
        "content": ai_reply
    })

    return ai_reply

# Start chatting
print("Aria is ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Aria: Goodbye! Keep building!")
        break

    if user_input.strip() == "":
        continue

    response = chat(user_input)
    print(f"\nAria: {response}\n")