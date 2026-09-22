"""Semantic, cross-session memory for Pydantic AI, backed by VectorAI DB.

Run once to create the collection:  python vectoraidb_memory_store.py
"""
import logging
import time
import uuid

from actian_vectorai import (
    Distance, Field, FilterBuilder, PointStruct, VectorAIClient, VectorParams,
)
from sentence_transformers import SentenceTransformer

log = logging.getLogger("memory")


class VectorAIDBStore:
    def __init__(self, url="localhost:6574", collection="agent_memory",
                 model="sentence-transformers/all-MiniLM-L6-v2", threshold=0.3):
        self.collection, self.threshold = collection, threshold
        self.model = SentenceTransformer(model)
        self.client = VectorAIClient(url).__enter__()  # closed in __exit__

        if not self.client.collections.exists(collection):
            dim = len(self._embed("dimension probe"))  # 384 for all-MiniLM-L6-v2
            self.client.collections.create(
                collection, vectors_config=VectorParams(size=dim, distance=Distance.Cosine)
            )

    def _embed(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def _filter(self, user_id, **fields):
        # Every read and delete is scoped to one user.
        fb = FilterBuilder().must(Field("user_id").eq(user_id))
        for key, value in fields.items():
            if value is not None:
                fb = fb.must(Field(key).eq(value))
        return fb.build()

    def store(self, content, *, user_id, session_id, memory_type="fact"):
        memory_id = uuid.uuid4()
        payload = {
            "memory_id": str(memory_id), "content": content, "user_id": user_id,
            "session_id": session_id, "memory_type": memory_type,
            "created_at": int(time.time()),
        }
        point = PointStruct(id=memory_id.int >> 65,  # point IDs are 63-bit integers
                            vector=self._embed(content), payload=payload)
        self.client.points.upsert(self.collection, [point])
        self.client.vde.flush(self.collection)  # on disk before the process exits
        log.info("store  [%s] %r", memory_type, content)
        return payload

    def search(self, query, *, user_id, limit=5):
        hits = self.client.points.search(
            self.collection, vector=self._embed(query), limit=limit,
            score_threshold=self.threshold, with_payload=True, filter=self._filter(user_id),
        ) or []
        log.info("search %r -> %d hits %s", query, len(hits), [round(h.score, 3) for h in hits])
        return [{**h.payload, "score": h.score} for h in hits]

    def list(self, *, user_id, session_id=None, limit=100):
        points, _ = self.client.points.scroll(
            self.collection, limit=limit, filter=self._filter(user_id, session_id=session_id),
            with_payload=True, with_vectors=False,
        )
        return [p.payload for p in points]

    def delete(self, *, user_id, memory_id=None, session_id=None):
        """Delete one memory, one session, or (with no filters) all of a user's memories."""
        flt = self._filter(user_id, memory_id=memory_id, session_id=session_id)
        count = self.client.points.count(self.collection, filter=flt)
        if count:
            self.client.points.delete(self.collection, filter=flt)
        log.info("delete %d memories for %s", count, user_id)
        return count

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.client.__exit__(*exc)


if __name__ == "__main__":
    with VectorAIDBStore() as store:
        print(f"Collection '{store.collection}' is ready")
