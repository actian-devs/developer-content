"""Ask one catalog question end to end: VectorAI DB retrieves the top 3 entries, llama-server answers.

Start llama-server with one of the picks first, then:

  uv run demo.py
  uv run demo.py "Which plant makes the Halcyon 70 gear housing?"
"""
import sys
import tempfile
from pathlib import Path

import requests

from qpick import prompts, stack

DEFAULT_Q = "What is the maximum load of the Cobalt 41 clamp collar?"   # one of the 20 test questions


def main():
    """Retrieve for one question, print the entries VectorAI DB returned, then print the model's answer."""
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_Q
    # A fresh cache folder, so the question is embedded and searched now instead of read from the test's cache
    with tempfile.TemporaryDirectory() as tmp:
        hit = stack.retrieve([question], Path(tmp))[0]
    print(f"Question: {question}\n\nVectorAI DB top {stack.TOP_K}:")
    for chunk in hit["chunks"]:
        print(f"  - {chunk}")

    # The same prompt and settings the test uses: temperature 0, a fixed seed, a short answer
    body = {"messages": prompts.rag_prompt(question, hit["chunks"]), "max_tokens": 96, "temperature": 0, "seed": 1}
    r = requests.post("http://127.0.0.1:8080/v1/chat/completions", json=body, timeout=180)
    r.raise_for_status()
    print(f"\nAnswer: {r.json()['choices'][0]['message']['content'].strip()}")


if __name__ == "__main__":
    main()
