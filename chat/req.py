import requests

url = "http://127.0.0.1:8000/chat"
payload = {"message": "tell me about AI regulation", "thread_id": "123"}
response = requests.post(url, json=payload)

print(response.json())