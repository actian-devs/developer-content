"""Session 2: a fresh process that never saw session 1."""
import uuid

from memory_agent import ask
from vectoraidb_memory_store import VectorAIDBStore

with VectorAIDBStore() as store:
    print("memories on disk:", len(store.list(user_id="user-42")))
    reply = ask(store, "user-42", uuid.uuid4().hex[:8],
                "What tools do I use for infrastructure deployments?")
    print("agent:", reply)
