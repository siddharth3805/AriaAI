from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# ── 1. FLASK SETUP ─────────────────────────────
app = Flask(__name__)
CORS(app)

# ── 2. AI SETUP ────────────────────────────────
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=500
)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Aria, an elite AI assistant.
Be helpful, concise, and friendly.
Always give practical and clear responses."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model | parser

# ── 3. SESSION MEMORY ──────────────────────────
# Stores conversation per session
sessions = {}

def get_history(session_id):
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]

# ── 4. ROUTES ──────────────────────────────────

# Health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "AriaAI API is live!",
        "endpoints": ["/chat", "/clear", "/history"]
    })

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        session_id = data.get("session_id", "default")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Get session history
        history = get_history(session_id)

        # Get AI response
        response = chain.invoke({
            "input": user_message,
            "history": history
        })

        # Save to history
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=response))

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "user_message": user_message,
            "response": response,
            "history_length": len(history)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Clear history
@app.route("/clear", methods=["POST"])
def clear():
    data = request.get_json()
    session_id = data.get("session_id", "default")

    if session_id in sessions:
        sessions[session_id] = []

    return jsonify({
        "status": "success",
        "message": f"History cleared for session {session_id}"
    })

# View history
@app.route("/history", methods=["GET"])
def history():
    session_id = request.args.get("session_id", "default")
    history = get_history(session_id)

    messages = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        else:
            messages.append({"role": "assistant", "content": msg.content})

    return jsonify({
        "session_id": session_id,
        "messages": messages,
        "total": len(messages)
    })

# ── 5. RUN ─────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   AriaAI Flask API Starting...")
    print("=" * 50)
    app.run(debug=True, port=5000)