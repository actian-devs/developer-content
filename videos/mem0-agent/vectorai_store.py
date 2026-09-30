"""A LangChain VectorStore over VectorAI DB — the surface mem0 calls."""

import os
import uuid

from actian_vectorai import VectorAIClient
from actian_vectorai.models import Distance, PointStruct, VectorParams
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

HOST = os.environ.get("VECTORAI_HOST", "localhost:6574")  # 6574 is gRPC; 6575 is the web UI


def connect(host=HOST):
    client = VectorAIClient(host)
    client.connect()
    return client


class VectorAIStore(VectorStore):
    def __init__(self, host=HOST, collection="agent_memory", dim=768):
        self.collection = collection
        self.client = connect(host)
        if self.client.collections.exists(collection):
            # collections stay on disk across a DB restart but aren't served until reopened
            self.client.vde.open_collection(collection)
        else:
            self.client.collections.create(
                collection,
                vectors_config=VectorParams(size=dim, distance=Distance.Cosine),
            )

    # mem0 has already embedded and passes no texts - never embed in here
    def add_embeddings(self, embeddings, metadatas=None, ids=None, texts=None, **kwargs):
        embeddings = list(embeddings)
        ids = list(ids) if ids else [uuid.uuid4().hex for _ in embeddings]
        metadatas = list(metadatas) if metadatas else [{} for _ in embeddings]
        points = [
            PointStruct(id=i, vector=list(v), payload=dict(m or {}))
            for i, v, m in zip(ids, embeddings, metadatas)
        ]
        self.client.points.upsert(self.collection, points=points)
        return ids

    def similarity_search_with_score_by_vector(self, embedding, k=5, **kwargs):
        hits = self.client.points.search(self.collection, vector=list(embedding))  # `vector=`, not `query_vector=`
        return [(self._document(h), float(h.score)) for h in hits[:k]]

    def similarity_search_by_vector(self, embedding, k=5, **kwargs):
        return [doc for doc, _ in self.similarity_search_with_score_by_vector(embedding, k)]

    # memory gets revised, so mem0 needs the store addressable by id
    def get_by_ids(self, ids):
        points = self.client.points.get(self.collection, ids=list(ids))
        return [self._document(p) for p in points]

    def delete(self, ids=None, **kwargs):
        if ids:
            self.client.points.delete(self.collection, points=list(ids))
        return True

    def _document(self, hit):
        payload = getattr(hit, "payload", None) or {}
        return Document(page_content=payload.get("data", ""), metadata=payload, id=str(hit.id))

    # required by the LangChain base class; mem0 never calls these
    def add_texts(self, texts, metadatas=None, **kwargs):
        raise NotImplementedError("mem0 supplies vectors - use add_embeddings")

    def similarity_search(self, query, k=5, **kwargs):
        raise NotImplementedError("mem0 searches by vector - use similarity_search_by_vector")

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        raise NotImplementedError
