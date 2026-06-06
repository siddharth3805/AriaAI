# ZyraNovaAI 🤖

> Production-grade AI Assistant built with Python, Groq, LangChain & Streamlit

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-green)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)

---

## 🚀 What is ZyraNovaAI?

ZyraNovaAI is a fully production-grade AI assistant built from 
scratch as part of an intensive AI Engineering bootcamp. It combines 
multiple AI technologies into one unified system powered by Zyra — 
your intelligent AI assistant.

---

## ✨ Features

| Feature | Technology | File |
|---|---|---|
| Memory Chatbot | Groq + LLaMA 3 | `zyra_chatbot.py` |
| Prompt Engineering | 6 techniques | `prompt_engineering.py` |
| RAG Pipeline | ChromaDB + Embeddings | `rag_basics.py` |
| PDF Question Answering | LangChain + ChromaDB | `pdf_qa.py` |
| Voice Assistant | Whisper + pyttsx3 | `voice_assistant.py` |
| REST API | Flask | `app.py` |
| Web UI | Streamlit | `streamlit_app.py` |

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **LLM:** LLaMA 3 via Groq API
- **Framework:** LangChain
- **Vector DB:** ChromaDB
- **Embeddings:** Sentence Transformers
- **Speech:** OpenAI Whisper + pyttsx3
- **Backend:** Flask REST API
- **Frontend:** Streamlit
- **Version Control:** Git + GitHub

---

## 📁 Project Structure
ZyraNovaAI/
├── zyra_chatbot.py          # Memory chatbot with LangChain
├── prompt_engineering.py    # 6 prompt engineering techniques
├── langchain_basics.py      # LangChain 4 building blocks
├── rag_basics.py            # RAG pipeline + ChromaDB
├── pdf_qa.py                # PDF Question Answering system
├── voice_assistant.py       # Voice AI with Whisper
├── app.py                   # Flask REST API
├── streamlit_app.py         # Streamlit Web UI
├── test_api.py              # API testing script
├── requirements.txt         # Dependencies
├── .env.example             # Environment template
└── README.md                # This file

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/siddharth3805/ZyraNovaAI.git
cd ZyraNovaAI
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file:

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
