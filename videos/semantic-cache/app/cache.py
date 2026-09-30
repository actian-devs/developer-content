"""The cache layer: embed the query, find the nearest previous one, serve it or fall through."""

import os
import sys
import threading
import time
import uuid

from actian_vectorai import VectorAIClient
from actian_vectorai.models import Distance, PointStruct, VectorParams
from fastembed import TextEmbedding

from app.llm import MODEL

HOST = os.environ.get("VECTORAI_HOST", "localhost:6574")  # 6574 is gRPC; 6575 is the web UI
COLLECTION = "llm_cache"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
THRESHOLD = 0.60
TTL_SECONDS = 24 * 60 * 60
PROMPT_VERSION = "v1"


def connect(host=HOST):
    client = VectorAIClient(host)
    client.connect()
    return client


def create_collection(client):
    client.collections.create(
        COLLECTION,
        # cosine ignores magnitude, so a terse question and a rambling one still match
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.Cosine),
    )


class SemanticCache:
    def __init__(self, host=HOST, threshold=THRESHOLD, model=MODEL):
        self.threshold = threshold
        self.model = model
        self.embed = TextEmbedding(EMBED_MODEL)
        self._lock = threading.Lock()  # the write-back runs on a background thread
        self.client = connect(host)
        if self.client.collections.exists(COLLECTION):
            # collections stay on disk across a DB restart but aren't served until reopened
            self.client.vde.open_collection(COLLECTION)
        else:
            create_collection(self.client)

    def vector(self, text):
        with self._lock:
            return next(self.embed.embed([text])).tolist()

    def lookup(self, query, scope=None):
        """Hot path - runs on every request, including the ones that miss."""
        hits = self.client.points.search(COLLECTION, vector=self.vector(query))  # `vector=`, not `query_vector=`
        if not hits:
            return None

        hit = hits[0]
        payload = getattr(hit, "payload", None) or {}
        score = float(hit.score)

        if score < self.threshold:
            return None
        if time.time() - payload.get("created_at", 0) > payload.get("ttl", TTL_SECONDS):
            return None
        # answers from a model or prompt we no longer run are not answers
        if payload.get("model") != self.model or payload.get("prompt_version") != PROMPT_VERSION:
            return None
        if payload.get("scope") != scope:
            return None

        return {"answer": payload.get("answer", ""), "score": score, "matched": payload.get("query", "")}

    def store(self, query, answer, scope=None):
        """Write-on-miss. Runs after the response has already gone back."""
        point = PointStruct(
            id=uuid.uuid4().hex,
            vector=self.vector(query),
            payload={
                "query": query,
                "answer": answer,
                "created_at": time.time(),
                "ttl": TTL_SECONDS,
                "model": self.model,
                "prompt_version": PROMPT_VERSION,
                "scope": scope,
            },
        )
        self.client.points.upsert(COLLECTION, points=[point])


if __name__ == "__main__":
    if "--init" in sys.argv:
        cache = SemanticCache()
        count = cache.client.points.count(COLLECTION)
        print(f"collection {COLLECTION} ready | {count} points")
