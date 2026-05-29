# models/ollama_client.py

import requests
from models.base import ModelClient
from config import OLLAMA_URL


class OllamaClient(ModelClient):
    def __init__(self, model_name="llama3", temperature=0.7):
        super().__init__(model_name, temperature)

    def generate(self, prompt):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False
            }
        )

        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.text}")

        return response.json()["response"]