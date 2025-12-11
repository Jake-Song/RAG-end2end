"""
Open WebUI Function for Agent
FASTAPI 서버 실행 후 (uvicorn chat.agent:app --reload) 연결
"""
from pydantic import BaseModel, Field
import requests


class Pipe:
    class Valves(BaseModel):
        MODEL_ID: str = Field(default="my-agent")

    def __init__(self):
        self.valves = self.Valves()

    def pipe(self, body: dict):
        headers = {
            "Content-Type": "application/json",
        }

        # Update the model id in the body
        thread_id = "open_webui_1"
        payload = {"messages": body["messages"], "thread_id": thread_id}
        print(f"body: {body}")
        try:
            response = requests.post("http://127.0.0.1:8000/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            print(f"body: {body}")
            print(f"data: {data}")

            return data["response"]

        except Exception as e:
            return f"Error: {e}"