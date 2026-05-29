# models/ollama_chat_client.py

import requests
from config import OLLAMA_URL


CHAT_URL = OLLAMA_URL.replace("/api/generate", "/api/chat")


class OllamaChatClient:
    """
    Wraps Ollama's /api/chat endpoint so conversation history is maintained
    across turns. Each call to `chat()` appends to the internal message history.
    """

    def __init__(self, model_name="llama3", temperature=0.7, system_prompt=None):
        self.model_name = model_name
        self.temperature = temperature
        self.messages = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_message: str) -> str:
        """Send a user message, get an assistant reply, and remember both."""
        self.messages.append({"role": "user", "content": user_message})

        response = requests.post(
            CHAT_URL,
            json={
                "model": self.model_name,
                "messages": self.messages,
                "options": {"temperature": self.temperature},
                "stream": False,
            },
        )

        if response.status_code != 200:
            raise Exception(f"Ollama chat error: {response.text}")

        reply = response.json()["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, system_prompt=None):
        """Clear history (and optionally set a new system prompt) for a fresh conversation."""
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
