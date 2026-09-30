"""mem0, configured to keep its memories in VectorAI DB."""

import quiet  # noqa: F401  - must come before mem0, see quiet.py

from mem0 import Memory

from vectorai_store import HOST, VectorAIStore

COLLECTION = "agent_memory"
EMBED_DIM = 768  # nomic-embed-text; the collection is created at this width too
LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
USER = "ada"


def build_memory(host=HOST):
    store = VectorAIStore(host=host, collection=COLLECTION, dim=EMBED_DIM)
    config = {
        "llm": {"provider": "ollama", "config": {"model": LLM_MODEL}},
        "embedder": {"provider": "ollama", "config": {"model": EMBED_MODEL}},
        # `langchain` is the one provider that takes a store object instead of a hostname
        "vector_store": {
            "provider": "langchain",
            "config": {"client": store, "collection_name": COLLECTION},
        },
    }
    return Memory.from_config(config)
