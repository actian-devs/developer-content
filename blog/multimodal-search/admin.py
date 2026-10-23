from __future__ import annotations

import sys

from actian_vectorai import VectorAIClient

action, name = sys.argv[1], sys.argv[2]
with VectorAIClient("localhost:6584") as client:
    if action == "count":
        info = client.collections.get_info(name)
        print(f"{info.vectors_count} vectors in '{name}'")
    elif action == "drop":
        client.collections.delete(name)
        print(f"dropped '{name}'")
    else:
        raise SystemExit("usage: admin.py count|drop <collection>")