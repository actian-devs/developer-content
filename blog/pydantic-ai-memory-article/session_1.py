"""Session 1: tell the agent something, then exit."""
import uuid

from memory_agent import ask
from vectoraidb_memory_store import VectorAIDBStore

with VectorAIDBStore() as store:
    reply = ask(store, "user-42", uuid.uuid4().hex[:8],
                "For future chats: our production EKS cluster runs in eu-west-1, "
                "and we deploy infrastructure with Terraform through GitHub Actions.")
    print("agent:", reply)
