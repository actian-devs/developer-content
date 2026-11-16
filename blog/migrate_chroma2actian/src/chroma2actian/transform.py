"""Pure conversion rules: Chroma record -> Actian point.

Everything here is deterministic and side-effect free, so the migrator and the validator
apply *exactly* the same mapping.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np
from actian_vectorai import Distance, PointStruct, VectorParams

from .errors import UsageError

# Fixed namespace for the default ID mapping. Changing it changes every mapped ID.
DEFAULT_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/chroma2actian/id-namespace/v1")

_INT_ID = re.compile(r"(0|[1-9][0-9]*)")
UINT64_MAX = (1 << 64) - 1                       # Actian numeric point IDs are uint64
INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1  # Actian payload integers are int64
MAX_DIMENSION = VectorParams.MAX_DIMENSION        # 8192 in SDK 1.0.x

SPACE_TO_DISTANCE = {"l2": Distance.Euclid, "cosine": Distance.Cosine, "ip": Distance.Dot}
DISTANCE_NAMES = {"cosine": Distance.Cosine, "euclid": Distance.Euclid, "dot": Distance.Dot}


class IdStrategy(str, Enum):
    UUID5 = "uuid5"        # every ID -> deterministic UUIDv5 (default; uniform, always valid)
    PRESERVE = "preserve"  # canonical UUIDs / non-negative ints kept as-is; everything else -> UUIDv5
    NUMERIC = "numeric"    # IDs must already be canonical non-negative integers (strict)


class BadRecord(Exception):
    """A single record cannot be converted; the policy (fail/skip) is applied by the caller."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- IDs
def map_id(chroma_id: str, strategy: IdStrategy = IdStrategy.UUID5,
           namespace: uuid.UUID = DEFAULT_UUID_NAMESPACE) -> int | str:
    """Map a Chroma string ID to a valid Actian point ID (non-negative int or UUID string)."""
    if not isinstance(chroma_id, str):
        raise BadRecord(f"Chroma ID is not a string: {type(chroma_id).__name__}")
    if strategy is IdStrategy.UUID5:
        return str(uuid.uuid5(namespace, chroma_id))
    if _INT_ID.fullmatch(chroma_id) and int(chroma_id) <= UINT64_MAX:
        return int(chroma_id)
    if strategy is IdStrategy.NUMERIC:
        raise BadRecord(f"ID {chroma_id!r} is not a canonical non-negative integer (--id-strategy numeric)")
    try:  # PRESERVE
        if str(uuid.UUID(chroma_id)) == chroma_id:
            return chroma_id
    except ValueError:
        pass
    return str(uuid.uuid5(namespace, chroma_id))


def canonical_id(actian_id: int | str) -> str:
    """Single string form for set comparison: decimal for ints, lowercase hyphenated for UUIDs."""
    if isinstance(actian_id, int):
        return str(actian_id)
    return str(uuid.UUID(str(actian_id)))


class IdSetDigest:
    """Order-independent, O(1)-memory fingerprint of a set of IDs.

        H(S) = ( |S| ,  sum_{id in S} h(id)  mod 2^128 ,  xor_{id in S} h(id) )     h = BLAKE2b-128

    Two equal sets always match; two different sets collide with probability ~2^-128.
    Hashing first matters: summing raw sequential IDs would let {1,4} and {2,3} collide.
    """

    _MASK = (1 << 128) - 1

    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self.xor = 0

    def add(self, canonical: str) -> None:
        h = int.from_bytes(hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).digest(), "big")
        self.count += 1
        self.total = (self.total + h) & self._MASK
        self.xor ^= h

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.count, self.total, self.xor)

    def hexdigest(self) -> str:
        return f"{self.count}:{self.total:032x}:{self.xor:032x}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IdSetDigest) and self.as_tuple() == other.as_tuple()

    __hash__ = None  # type: ignore[assignment]


def sample_bucket(chroma_id: str) -> int:
    """Stable pseudo-random 32-bit value per ID: lets validation sample the same records on every run."""
    return int.from_bytes(hashlib.blake2b(chroma_id.encode("utf-8"), digest_size=4).digest(), "big")


# ------------------------------------------------------------------ distance
def resolve_distance(space: str, override: str | None) -> Distance:
    if override and override != "auto":
        try:
            return DISTANCE_NAMES[override]
        except KeyError:
            raise UsageError(f"--distance must be auto|cosine|euclid|dot, got {override!r}") from None
    try:
        return SPACE_TO_DISTANCE[space]
    except KeyError:
        raise UsageError(f"Chroma distance space {space!r} has no Actian equivalent; pass --distance explicitly") from None


# ------------------------------------------------------------------- records
@dataclass
class TransformOptions:
    id_strategy: IdStrategy = IdStrategy.UUID5
    uuid_namespace: uuid.UUID = DEFAULT_UUID_NAMESPACE
    id_field: str = "chroma_id"          # payload key that keeps the original Chroma ID
    document_field: str = "document"     # payload key for Chroma's `documents`
    include_documents: bool = True
    include_uris: bool = False
    uri_field: str = "uri"
    reject_zero_vectors: bool = False    # set by the caller when the metric is cosine (zero norm is undefined)

    def fingerprint(self) -> dict[str, Any]:
        return {"id_strategy": self.id_strategy.value, "uuid_namespace": str(self.uuid_namespace),
                "id_field": self.id_field, "document_field": self.document_field,
                "include_documents": self.include_documents, "include_uris": self.include_uris,
                "uri_field": self.uri_field}


@dataclass
class Reject:
    chroma_id: str
    reason: str
    offset: int | None = None


@dataclass
class ConvertedPage:
    points: list[PointStruct] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)       # rough wire-size estimate per point (bytes)
    rejects: list[Reject] = field(default_factory=list)


def _approx_size(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(len(str(k)) + _approx_size(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_approx_size(v) for v in value)
    return 8


class RecordTransformer:
    def __init__(self, opts: TransformOptions, dimension: int | None = None) -> None:
        self.opts = opts
        self.dimension = dimension
        self.reserved = {opts.id_field}
        if opts.include_documents:
            self.reserved.add(opts.document_field)
        if opts.include_uris:
            self.reserved.add(opts.uri_field)
        self.field_map: dict[str, str] = {}          # original metadata key -> payload key (renames only)
        self.counters: Counter[str] = Counter()

    # -- values
    def _value(self, v: Any, key: str) -> Any:
        if v is None or isinstance(v, (bool, str)):
            return v
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.integer):
            v = int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, int):
            if INT64_MIN <= v <= INT64_MAX:
                return v
            self.counters["int_out_of_int64_stringified"] += 1
            return str(v)
        if isinstance(v, float):
            return v
        if isinstance(v, (list, tuple, np.ndarray)):
            return [self._value(x, key) for x in v]
        if isinstance(v, dict):
            return {str(k): self._value(x, key) for k, x in v.items()}
        raise BadRecord(f"unsupported metadata value type {type(v).__name__} for key {key!r}")

    def _key(self, k: str) -> str:
        if k in self.reserved:
            nk = f"meta_{k}"
            self.field_map[k] = nk
            self.counters["metadata_key_renamed"] += 1
            return nk
        return k

    # -- record
    def convert(self, chroma_id: str, embedding: Any, document: str | None, metadata: dict | None,
                uri: str | None = None) -> tuple[PointStruct, int]:
        pid = map_id(chroma_id, self.opts.id_strategy, self.opts.uuid_namespace)
        if embedding is None:
            raise BadRecord("record has no embedding")
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.ndim != 1 or vec.size == 0:
            raise BadRecord("embedding is not a non-empty 1-D vector")
        if self.dimension is not None and vec.shape[0] != self.dimension:
            raise BadRecord(f"embedding dimension {vec.shape[0]} != collection dimension {self.dimension}")
        if not np.isfinite(vec).all():
            raise BadRecord("embedding contains NaN or Inf")
        if self.opts.reject_zero_vectors and not vec.any():
            raise BadRecord("zero vector (undefined under cosine distance; pass --allow-zero-vectors to keep)")

        payload: dict[str, Any] = {}
        for k, v in (metadata or {}).items():
            if not isinstance(k, str):
                raise BadRecord(f"metadata key {k!r} is not a string")
            nk = self._key(k)
            if nk in payload:
                raise BadRecord(f"metadata key {k!r} collides with another key after renaming to {nk!r}")
            payload[nk] = self._value(v, k)
        payload[self.opts.id_field] = chroma_id
        if self.opts.include_documents and document is not None:
            payload[self.opts.document_field] = document
        elif self.opts.include_documents:
            self.counters["document_none_omitted"] += 1
        if self.opts.include_uris and uri is not None:
            payload[self.opts.uri_field] = uri
        try:
            point = PointStruct(id=pid, vector=vec.tolist(), payload=payload)
        except Exception as exc:  # pydantic validation of IDs/vectors
            raise BadRecord(f"SDK rejected the point: {exc}") from exc
        return point, int(vec.nbytes) + _approx_size(payload)

    def convert_page(self, page: Any) -> ConvertedPage:
        out = ConvertedPage()
        n = len(page.ids)
        for i in range(n):
            cid = page.ids[i]
            try:
                point, size = self.convert(
                    cid,
                    page.embeddings[i] if page.embeddings is not None else None,
                    page.documents[i] if page.documents is not None else None,
                    page.metadatas[i] if page.metadatas is not None else None,
                    page.uris[i] if getattr(page, "uris", None) is not None else None,
                )
            except BadRecord as bad:
                out.rejects.append(Reject(str(cid), bad.reason, page.offset + i))
                continue
            out.points.append(point)
            out.sizes.append(size)
        return out


def chunk_points(points: list[PointStruct], sizes: list[int], max_count: int, max_bytes: int) -> Iterable[list[PointStruct]]:
    """Split into batches bounded by both record count and estimated bytes (a single oversize point still goes alone)."""
    batch: list[PointStruct] = []
    acc = 0
    for p, s in zip(points, sizes):
        if batch and (len(batch) >= max_count or acc + s > max_bytes):
            yield batch
            batch, acc = [], 0
        batch.append(p)
        acc += s
    if batch:
        yield batch
