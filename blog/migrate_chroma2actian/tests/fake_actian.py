"""In-memory stand-in for ``actian_vectorai.VectorAIClient``.

It deliberately uses the SDK's *real* models and exceptions, and serialises every upserted point with the
SDK's own ``to_proto()`` so payload/ID/vector encoding problems surface exactly as they would on the wire.
What it does NOT prove is server behaviour: see tests/test_live_actian.py for the opt-in live smoke test.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

import numpy as np
from actian_vectorai import (CapacityExceededError, CollectionExistsError, CollectionNotFoundError, CollectionInfo,
                             CollectionState, Distance, MessageSizeExceededError, PointStruct, RetrievedPoint,
                             ScoredPoint, UpdateResult, UpdateStatus, VectorParams)
from actian_vectorai.models.collections import CollectionConfig, CollectionParams


def _sort_key(pid: int | str) -> tuple[int, Any]:
    return (0, pid) if isinstance(pid, int) else (1, str(pid))


class _Coll:
    def __init__(self, size: int, distance: Distance) -> None:
        self.size, self.distance = size, distance
        self.rows: dict[int | str, tuple[np.ndarray, dict[str, Any]]] = {}


class FakeVectorAI:
    def __init__(self, *, max_message_points: int | None = None, max_points: int | None = None,
                 normalize_cosine: bool = True) -> None:
        self.data: dict[str, _Coll] = {}
        self.max_message_points = max_message_points
        self.max_points = max_points
        self.normalize_cosine = normalize_cosine
        self.upsert_calls = 0
        self.on_upsert: Callable[[list[PointStruct]], None] | None = None   # fault-injection hook
        self.collections = _Collections(self)
        self.points = _Points(self)
        self.vde = _Vde(self)
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def health_check(self) -> dict[str, Any]:
        return {"title": "Fake Actian VectorAI DB", "version": "test"}


class _Collections:
    def __init__(self, s: FakeVectorAI) -> None:
        self.s = s

    def exists(self, name: str) -> bool:
        return name in self.s.data

    def create(self, name: str, *, vectors_config: VectorParams, **kwargs: Any) -> bool:
        if name in self.s.data:
            raise CollectionExistsError(f"{name} exists")
        self.s.data[name] = _Coll(vectors_config.size, vectors_config.distance)
        return True

    def delete(self, name: str, **kwargs: Any) -> bool:
        self.s.data.pop(name, None)
        return True

    def get_info(self, name: str, **kwargs: Any) -> CollectionInfo:
        c = self.s.data.get(name)
        if c is None:
            raise CollectionNotFoundError(name)
        return CollectionInfo(points_count=len(c.rows), config=CollectionConfig(
            params=CollectionParams(vectors_config=VectorParams(size=c.size, distance=c.distance))))


class _Points:
    def __init__(self, s: FakeVectorAI) -> None:
        self.s = s

    def _coll(self, name: str) -> _Coll:
        try:
            return self.s.data[name]
        except KeyError:
            raise CollectionNotFoundError(name) from None

    def upsert(self, name: str, points: list[PointStruct], **kwargs: Any) -> UpdateResult:
        c = self._coll(name)
        self.s.upsert_calls += 1
        if self.s.on_upsert:
            self.s.on_upsert(points)
        if self.s.max_message_points is not None and len(points) > self.s.max_message_points:
            raise MessageSizeExceededError(f"{len(points)} points exceed the message limit")
        for p in points:
            p.to_proto()                                  # real SDK serialisation (raises on bad payloads)
        new = sum(1 for p in points if p.id not in c.rows)
        total = sum(len(x.rows) for x in self.s.data.values())
        if self.s.max_points is not None and total + new > self.s.max_points:
            raise CapacityExceededError("licence limit reached")
        for p in points:
            v = np.asarray(p.vector, dtype=np.float32)
            if v.shape[0] != c.size:
                raise ValueError(f"dimension {v.shape[0]} != {c.size}")
            if c.distance == Distance.Cosine and self.s.normalize_cosine and v.any():
                v = v / np.linalg.norm(v)
            c.rows[p.id] = (v, dict(p.payload or {}))
        return UpdateResult(status=UpdateStatus.Completed)

    def count(self, name: str, **kwargs: Any) -> int:
        return len(self._coll(name).rows)

    def scroll(self, name: str, *, offset: Any = None, limit: int | None = None, with_payload: bool = True,
               with_vectors: bool = False, **kwargs: Any) -> tuple[list[RetrievedPoint], Any]:
        c = self._coll(name)
        ids = sorted(c.rows, key=_sort_key)
        start = 0 if offset is None else ids.index(offset)        # inclusive-offset semantics
        page = ids[start:start + (limit or 10)]
        nxt = ids[start + len(page)] if start + len(page) < len(ids) else None
        out = [RetrievedPoint(id=i, payload=c.rows[i][1] if with_payload else None,
                              vectors=c.rows[i][0].tolist() if with_vectors else None) for i in page]
        return out, nxt

    def get(self, name: str, ids: list[Any], *, with_payload: bool = True, with_vectors: bool = False,
            **kwargs: Any) -> list[RetrievedPoint]:
        c = self._coll(name)
        norm = [i if isinstance(i, int) else str(uuid.UUID(str(i))) for i in ids]
        return [RetrievedPoint(id=i, payload=c.rows[i][1] if with_payload else None,
                               vectors=c.rows[i][0].tolist() if with_vectors else None)
                for i in norm if i in c.rows]

    def search(self, name: str, vector: Any, *, limit: int = 10, **kwargs: Any) -> list[ScoredPoint]:
        c = self._coll(name)
        q = np.asarray(vector, dtype=np.float32)
        ids = list(c.rows)
        if not ids:
            return []
        m = np.stack([c.rows[i][0] for i in ids])
        if c.distance == Distance.Euclid:
            scores = -np.sum((m - q) ** 2, axis=1)
        elif c.distance == Distance.Dot:
            scores = m @ q
        else:
            qn = q / (np.linalg.norm(q) or 1.0)
            scores = (m @ qn) / np.where(np.linalg.norm(m, axis=1) == 0, 1.0, np.linalg.norm(m, axis=1))
        order = np.argsort(-scores, kind="stable")[:limit]
        return [ScoredPoint(id=ids[i], score=float(scores[i])) for i in order]


class _Vde:
    def __init__(self, s: FakeVectorAI) -> None:
        self.s = s
        self.flushed: list[str] = []

    def flush(self, name: str, **kwargs: Any) -> bool:
        self.flushed.append(name)
        return True

    def wait_for_ready(self, name: str, **kwargs: Any) -> CollectionState:
        return CollectionState.READY
