"""The retrieval stack that shares the board with the LLM: Ollama embeddings and VectorAI DB.

The catalog is embedded once with nomic-embed-text and stored in an HNSW collection. Both stay
loaded for every test, so each memory reading includes the retrieval stack, not the LLM alone.
"""
import json
from pathlib import Path

import numpy as np
import requests
from actian_vectorai import Distance, PointStruct, VectorAIClient, VectorParams

EMBED_MODEL = "nomic-embed-text"       # the embedding model Ollama serves
OLLAMA = "http://localhost:11434"      # Ollama's HTTP address on the Pi
VECTORAI = "localhost:6574"            # VectorAI DB's gRPC port, which the Python SDK uses
COLLECTION = "parts_catalog"           # the VectorAI DB collection that holds the catalog
TOP_K = 3                              # how many catalog entries each RAG question gets


def embed(texts: list[str], prefix: str) -> np.ndarray:
    """nomic-embed-text expects a task prefix: search_document for chunks, search_query for questions."""
    out = []
    # Send the texts to Ollama 64 at a time; keep_alive keeps the model loaded for 24 hours
    for i in range(0, len(texts), 64):
        r = requests.post(f"{OLLAMA}/api/embed", timeout=600, json={
            "model": EMBED_MODEL, "input": [f"{prefix}: {t}" for t in texts[i:i + 64]], "keep_alive": "24h"})
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    v = np.asarray(out, dtype=np.float32)
    # Scale every vector to length 1, so cosine similarity is a plain dot product
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_index(recs: list[dict], cache: Path) -> None:
    """Embed the catalog (cached on disk) and load it into VectorAI DB."""
    cache.mkdir(parents=True, exist_ok=True)
    vec_file = cache / f"catalog-{EMBED_MODEL}-{len(recs)}.npy"
    # Embedding 4,000 entries takes a while on a Pi, so the vectors are saved and reused on later runs
    if vec_file.exists():
        vectors = np.load(vec_file)
    else:
        vectors = embed([r["text"] for r in recs], "search_document")
        np.save(vec_file, vectors)
    with VectorAIClient(VECTORAI) as db:
        # Start from an empty collection, so a rerun never adds the catalog twice
        if db.collections.exists(COLLECTION):
            db.collections.delete(COLLECTION)
        db.collections.create(COLLECTION, vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.Cosine))
        # One point per catalog entry: its vector, with the entry's text stored alongside it
        points = [PointStruct(id=r["id"], vector=v.tolist(), payload={"text": r["text"]}) for r, v in zip(recs, vectors)]
        for i in range(0, len(points), 500):   # upload in batches of 500
            db.points.upsert(COLLECTION, points[i:i + 500])


def retrieve(questions: list[str], cache: Path) -> list[dict]:
    """Top-K chunks for every RAG question. Retrieval does not depend on the LLM, so it runs once and is
    cached: every model and level answers from exactly the same chunks."""
    path = cache / "retrieved.json"
    if path.exists():
        return json.loads(path.read_text())
    qv = embed(questions, "search_query")
    # One request to VectorAI DB searches for all 20 questions at once
    with VectorAIClient(VECTORAI) as db:
        hits = db.points.search_batch(COLLECTION, [{"vector": v.tolist(), "limit": TOP_K, "with_payload": True}
                                                   for v in qv])
    # Keep the IDs (to check retrieval) and the text (to put in the prompt) of the top entries
    out = [{"ids": [int(h.id) for h in hs], "chunks": [h.payload["text"] for h in hs]} for hs in hits]
    path.write_text(json.dumps(out, indent=1))
    return out


def warm_embedder() -> None:
    """Load the embedding model and keep it loaded, so its memory counts in every reading."""
    embed(["warm up"], "search_query")
