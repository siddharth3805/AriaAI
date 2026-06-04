from dotenv import load_dotenv
import os
import json

load_dotenv()

# 1. Create a message
def create_message(role, content):
    return {"role": role, "content": content}

# 2. Build conversation history
history =[]
history.append(create_message("user", "Hello!"))
history.append(create_message("assistant", "Hi! How can I help?"))
history.append(create_message("user", "What is AI?"))

# 3. Print it as JSON
print(json.dumps(history, indent=2))

# 4. Print each message
for msg in history:
    print(f"{msg['role'].upper()}: {msg['content']}")