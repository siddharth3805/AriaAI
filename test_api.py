import requests

BASE_URL = "http://127.0.0.1:5000"

# Test 1 — Health check
print("=== HEALTH CHECK ===")
response = requests.get(f"{BASE_URL}/")
print(response.json())

# Test 2 — Send message
print("\n=== CHAT TEST ===")
response = requests.post(
    f"{BASE_URL}/chat",
    json={
        "message": "What is LangChain?",
        "session_id": "test_session"
    }
)
print(response.json())

# Test 3 — Memory test
print("\n=== MEMORY TEST ===")
requests.post(
    f"{BASE_URL}/chat",
    json={"message": "My name is Siddharth", "session_id": "test_session"}
)
response = requests.post(
    f"{BASE_URL}/chat",
    json={"message": "What is my name?", "session_id": "test_session"}
)
print(response.json())

# Test 4 — View history
print("\n=== HISTORY ===")
response = requests.get(f"{BASE_URL}/history?session_id=test_session")
print(response.json())