"""The model call the cache is standing in front of."""

import os

import httpx

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
MODEL = os.environ.get("LLM_MODEL", "llama3.2")
SYSTEM = "Answer the support question in two or three sentences."


def complete(question, model=MODEL):
    response = httpx.post(
        OLLAMA,
        json={"model": model, "prompt": f"{SYSTEM}\n\nQ: {question}\nA:", "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
