"""
core/local_llm.py

Wrapper around Ollama local LLM.

This module is intentionally simple:
- No agent framework here.
- Just a reasoning layer.
- Keeps prompt engineering explicit and inspectable.
"""

import requests


class LocalLLM:

    def __init__(self, model: str = "mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:
        """
        Sends prompt to Ollama and returns generated response.
        """
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        response.raise_for_status()

        return response.json()["response"]