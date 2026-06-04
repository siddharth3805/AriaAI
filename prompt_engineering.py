from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(system_prompt, user_message):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content

# Same question — 3 different system prompts
question = "What is machine learning?"

# Persona 1 — Expert tutor
print("=== EXPERT TUTOR ===")
print(ask("You are an expert AI tutor. Explain concepts deeply with examples.", question))

# Persona 2 — Simple explainer
print("\n=== SIMPLE EXPLAINER ===")
print(ask("You are explaining to a 10 year old child. Use very simple words and fun examples.", question))

# Persona 3 — Interview coach
print("\n=== INTERVIEW COACH ===")
print(ask("You are an interview coach. Give answers in the exact format expected in a technical job interview.", question))

# ============ TECHNIQUE 2 — FEW SHOT ============
print("\n=== FEW SHOT — SENTIMENT ANALYSIS ===")

few_shot_prompt = """Classify the sentiment. Reply with ONE word only: POSITIVE, NEGATIVE, or NEUTRAL. No explanation. No punctuation.

Message: "This product is amazing!" → POSITIVE
Message: "Worst experience ever" → NEGATIVE
Message: "Package arrived today" → NEUTRAL
Message: "I love how fast the delivery was but the product broke immediately" →"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": few_shot_prompt}],
    max_tokens=5,
    temperature=0
)
print(response.choices[0].message.content)

# ============ TECHNIQUE 3 — CHAIN OF THOUGHT ============
print("\n=== CHAIN OF THOUGHT ===")

cot_prompt = """
A store sells apples for $2 each. Buy 3 get 1 free. How much for 8 apples?

Think step by step:
1. First figure out how many free apples I get
2. Then calculate how many I actually pay for
3. Then calculate total cost
"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": cot_prompt}],
    max_tokens=300,
    temperature=0
)
print(response.choices[0].message.content)


# ============ TECHNIQUE 4 — OUTPUT FORMATTING ============
print("\n=== OUTPUT FORMATTING — JSON EXTRACTION ===")

format_prompt = """
Extract information and return ONLY a JSON object.
No explanation. No extra text. Just JSON.

Text: "Siddharth is a 20 year old student from Pune learning AI engineering"

Return exactly:
{
  "name": "",
  "age": 0,
  "city": "",
  "learning": ""
}
"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": format_prompt}],
    max_tokens=100,
    temperature=0
)
print(response.choices[0].message.content)


# ============ TECHNIQUE 5 — ROLE PROMPTING ============
print("\n=== ROLE PROMPTING — CTO ADVICE ===")

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "You are a CTO of a successful AI startup. Give technical advice concisely in max 4 lines."
        },
        {
            "role": "user",
            "content": "Should I use Flask or FastAPI for my AI chatbot backend?"
        }
    ],
    max_tokens=300,
    temperature=0.7
)
print(response.choices[0].message.content)


# ============ TECHNIQUE 6 — CONSTRAINED PROMPTING ============
print("\n=== CONSTRAINED PROMPTING ===")

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Always follow the user's constraints exactly."
        },
        {
            "role": "user",
            "content": "Explain what an API is in exactly 3 sentences. Use a restaurant analogy. No technical jargon. End with one practical example."
        }
    ],
    max_tokens=300,
    temperature=0.7
)
print(response.choices[0].message.content)