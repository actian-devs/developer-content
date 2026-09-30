"""Search the memories from a fresh process. Nothing is carried over but the database."""

import sys

from memory import USER, build_memory

question = " ".join(sys.argv[1:]) or "which database am I running"

m = build_memory()

for hit in m.search(question, filters={"user_id": USER}).get("results", []):
    print(f"{hit['score']:.3f}  {hit['memory']}")
