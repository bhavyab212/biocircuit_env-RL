import http.client
import json
import os

from dotenv import load_dotenv

load_dotenv()

conn = http.client.HTTPSConnection("api.groq.com")
conn.request("POST", "/openai/v1/chat/completions",
    body=json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 5
    }),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('API_KEY')}"
    }
)
res = conn.getresponse()
print(res.status, res.read().decode())