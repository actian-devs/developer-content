"""The retrieval stack that shares the board with the LLM: Ollama embeddings and VectorAI DB.

The catalog is embedded once with nomic-embed-text and stored in an HNSW collection. Both stay
loaded for every test, so each memory reading includes the retrieval stack, not the LLM alone.
"""
import json
from pathlib import Path

import numpy as np
import requests
from actian_vectorai import Distance, PointStruct, VectorAIClient, VectorParams

EMBED_MODEL = "nomic-embed-text"
OLLAMA = "http://localhost:11434"
VECTORAI = "localhost:6574"
COLLECTION = "parts_catalog"
TOP_K = 3


def embed(texts: list[str], prefix: str) -> np.ndarray:
    """nomic-embed-text expects a task prefix: search_document for chunks, search_query for questions."""
    out = []
    for i in range(0, len(texts), 64):
        r = requests.post(f"{OLLAMA}/api/embed", timeout=600, json={
            "model": EMBED_MODEL, "input": [f"{prefix}: {t}" for t in texts[i:i + 64]], "keep_alive": "24h"})
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    v = np.asarray(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_index(recs: list[dict], cache: Path) -> None:
    """Embed the catalog (cached on disk) and load it into VectorAI DB."""
    cache.mkdir(parents=True, exist_ok=True)
    vec_file = cache / f"catalog-{EMBED_MODEL}-{len(recs)}.npy"
    if vec_file.exists():
        vectors = np.load(vec_file)
    else:
        vectors = embed([r["text"] for r in recs], "search_document")
        np.save(vec_file, vectors)
    with VectorAIClient(VECTORAI) as db:
        if db.collections.exists(COLLECTION):
            db.collections.delete(COLLECTION)
        db.collections.create(COLLECTION, vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.Cosine))
        points = [PointStruct(id=r["id"], vector=v.tolist(), payload={"text": r["text"]}) for r, v in zip(recs, vectors)]
        for i in range(0, len(points), 500):
            db.points.upsert(COLLECTION, points[i:i + 500])


def retrieve(questions: list[str], cache: Path) -> list[dict]:
    """Top-K chunks for every RAG question. Retrieval does not depend on the LLM, so it runs once and is
    cached: every model and level answers from exactly the same chunks."""
    path = cache / "retrieved.json"
    if path.exists():
        return json.loads(path.read_text())
    qv = embed(questions, "search_query")
    with VectorAIClient(VECTORAI) as db:
        hits = db.points.search_batch(COLLECTION, [{"vector": v.tolist(), "limit": TOP_K, "with_payload": True}
                                                   for v in qv])
    out = [{"ids": [int(h.id) for h in hs], "chunks": [h.payload["text"] for h in hs]} for hs in hits]
    path.write_text(json.dumps(out, indent=1))
    return out


def warm_embedder() -> None:
    """Load the embedding model and keep it loaded, so its memory counts in every reading."""
    embed(["warm up"], "search_query")
