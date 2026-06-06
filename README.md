Get free Groq API key at: https://console.groq.com

### 5. Run the app

**Streamlit UI (recommended):**
```bash
streamlit run streamlit_app.py
```

**Flask API:**
```bash
python app.py
```

**Memory Chatbot:**
```bash
python zyra_chatbot.py
```

**Voice Assistant:**
```bash
python voice_assistant.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/chat` | Send message, get response |
| POST | `/clear` | Clear conversation history |
| GET | `/history` | View conversation history |

### Example API call:
```python
import requests

response = requests.post(
    "http://localhost:5000/chat",
    json={
        "message": "What is machine learning?",
        "session_id": "user123"
    }
)
print(response.json())
```

---

## 🧠 AI Concepts Implemented

- **Prompt Engineering** — 6 techniques including CoT, Few-shot, Role prompting
- **RAG** — Retrieval Augmented Generation with vector similarity search
- **Embeddings** — Sentence transformers converting text to 384-dimensional vectors
- **Memory** — Conversation history management across sessions
- **Voice AI** — Speech-to-text with Whisper, text-to-speech with pyttsx3
- **LangChain LCEL** — Prompt | Model | Parser pipeline

---

## 📊 Roadmap

-  Python for AI fundamentals
-  Groq API integration
-  Memory chatbot
-  Prompt engineering (6 techniques)
-  LangChain integration
-  RAG pipeline
-  PDF Question Answering
-  Voice Assistant
-  Flask REST API
-  Streamlit Web UI
-  Deploy on Streamlit Cloud
-  Multi-document RAG
-  Agent system

---

## 👨‍💻 Author

**Siddharth Sonawane**
- AI Engineering Student
- Building production-grade AI systems
- GitHub: [@siddharth3805](https://github.com/siddharth3805)

---

## ⭐ Show your support

Give a ⭐ if this project helped you learn AI engineering!

---

*Built with ❤️ using Python, Groq, LangChain, and Streamlit*
