"""Validated local codebase memory with in-memory and Actian REST backends."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

COLLECTION_NAME = "astra_codebase_memory"
EMBEDDING_DIMENSION = 384


class MemoryValidationError(ValueError):
    """Raised when a memory or tool argument violates the memory contract."""


class MemoryBackendError(RuntimeError):
    """Raised when a storage backend cannot complete a requested operation."""


class CollectionSafetyError(MemoryBackendError):
    """Raised rather than altering an unexpected or incompatible collection."""


class UnconfirmedOutcomeError(MemoryValidationError):
    """Raised when code attempts to store a fix before its outcome is confirmed."""


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validated_embedding(value: object) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MemoryValidationError("embedding must be a numeric sequence")
    if len(value) != EMBEDDING_DIMENSION:
        raise MemoryValidationError(f"embedding must contain {EMBEDDING_DIMENSION} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise MemoryValidationError("embedding values must be finite numbers")
        converted = float(item)
        if not math.isfinite(converted):
            raise MemoryValidationError("embedding values must be finite numbers")
        result.append(converted)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One confirmed debugging result and its local embedding vector."""

    memory_id: str
    fix_description: str
    error_type: str
    outcome: str
    file_paths: tuple[str, ...]
    timestamp: str
    session_id: str
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in (
            "memory_id",
            "fix_description",
            "error_type",
            "outcome",
            "session_id",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if not isinstance(self.file_paths, tuple) or not self.file_paths:
            raise MemoryValidationError("file_paths must be a non-empty tuple")
        clean_paths = tuple(_required_text(path, "file path") for path in self.file_paths)
        object.__setattr__(self, "file_paths", clean_paths)
        timestamp = _required_text(self.timestamp, "timestamp")
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise MemoryValidationError("timestamp must be valid ISO 8601") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise MemoryValidationError("timestamp must include a timezone")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "embedding", _validated_embedding(self.embedding))

    @property
    def payload(self) -> dict[str, object]:
        """Return the complete non-vector payload stored beside the embedding."""

        return {
            "memory_id": self.memory_id,
            "fix_description": self.fix_description,
            "error_type": self.error_type,
            "outcome": self.outcome,
            "file_paths": list(self.file_paths),
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }

    @classmethod
    def from_storage(
        cls, payload: object, embedding: object, *, fallback_id: object = None
    ) -> MemoryRecord:
        """Validate and reconstruct a record returned by a backend."""

        if not isinstance(payload, Mapping):
            raise MemoryValidationError("stored payload must be an object")
        raw_paths = payload.get("file_paths")
        if not isinstance(raw_paths, list):
            raise MemoryValidationError("stored file_paths must be an array")
        memory_id = payload.get("memory_id", fallback_id)
        return cls(
            memory_id=memory_id,  # type: ignore[arg-type]
            fix_description=payload.get("fix_description"),  # type: ignore[arg-type]
            error_type=payload.get("error_type"),  # type: ignore[arg-type]
            outcome=payload.get("outcome"),  # type: ignore[arg-type]
            file_paths=tuple(raw_paths),
            timestamp=payload.get("timestamp"),  # type: ignore[arg-type]
            session_id=payload.get("session_id"),  # type: ignore[arg-type]
            embedding=embedding,  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """A validated record and its cosine-similarity score."""

    record: MemoryRecord
    score: float

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise MemoryValidationError("score must be a finite number")
        if not math.isfinite(float(self.score)):
            raise MemoryValidationError("score must be a finite number")
        object.__setattr__(self, "score", float(self.score))


class Embedder(Protocol):
    async def embed(self, text: str) -> tuple[float, ...]:
        """Return one normalized 384-dimensional vector."""


class MemoryBackend(Protocol):
    async def upsert(self, record: MemoryRecord) -> None: ...

    async def search(
        self, embedding: tuple[float, ...], *, top_k: int, error_type: str | None
    ) -> list[MemorySearchResult]: ...

    async def delete(self, memory_id: str) -> None: ...


class SentenceTransformerEmbedder:
    """Lazy local MiniLM embedder pinned to CPU execution."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self._model_factory = model_factory
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            factory = self._model_factory
            if factory is None:
                from sentence_transformers import SentenceTransformer

                factory = SentenceTransformer
            self._model = factory(self.model_name, device="cpu")
        return self._model

    def _embed_sync(self, text: str) -> tuple[float, ...]:
        clean_text = _required_text(text, "text")
        encoded = self._load_model().encode(
            clean_text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vector = _validated_embedding(encoded.tolist() if hasattr(encoded, "tolist") else encoded)
        norm = math.sqrt(sum(component * component for component in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise MemoryValidationError("embedding output must be normalized")
        return vector

    async def embed(self, text: str) -> tuple[float, ...]:
        return await asyncio.to_thread(self._embed_sync, text)


class InMemoryMemoryBackend:
    """Deterministic backend for unit tests and offline orchestration tests."""

    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}

    async def upsert(self, record: MemoryRecord) -> None:
        if not isinstance(record, MemoryRecord):
            raise MemoryValidationError("record must be a MemoryRecord")
        self.records[record.memory_id] = record

    async def search(
        self, embedding: tuple[float, ...], *, top_k: int, error_type: str | None
    ) -> list[MemorySearchResult]:
        query = _validated_embedding(embedding)
        _validate_search_arguments(top_k, error_type)
        query_norm = math.sqrt(sum(value * value for value in query))
        if query_norm == 0:
            raise MemoryValidationError("search embedding cannot be a zero vector")
        results: list[MemorySearchResult] = []
        for record in self.records.values():
            if error_type is not None and record.error_type != error_type:
                continue
            record_norm = math.sqrt(sum(value * value for value in record.embedding))
            score = sum(a * b for a, b in zip(query, record.embedding, strict=True))
            score /= query_norm * record_norm
            results.append(MemorySearchResult(record=record, score=score))
        return sorted(results, key=lambda item: (-item.score, item.record.memory_id))[:top_k]

    async def delete(self, memory_id: str) -> None:
        self.records.pop(_required_text(memory_id, "memory_id"), None)


def _validate_search_arguments(top_k: int, error_type: str | None) -> None:
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise MemoryValidationError("top_k must be an integer between 1 and 10")
    if error_type is not None:
        _required_text(error_type, "error_type")


class ActianVectorAIBackend:
    """Actian VectorAI DB backend using the client's asynchronous REST transport."""

    def __init__(
        self,
        base_url: str,
        *,
        collection_name: str = COLLECTION_NAME,
        grpc_url: str = "localhost:16574",
        transport_factory: Callable[[str], Any] | None = None,
        collection_opener: Callable[[str, str], Awaitable[bool]] | None = None,
    ) -> None:
        if collection_name != COLLECTION_NAME:
            raise CollectionSafetyError(f"only {COLLECTION_NAME!r} is allowed")
        self.base_url = _required_text(base_url, "base_url").rstrip("/")
        self.grpc_url = _required_text(grpc_url, "grpc_url")
        self.collection_name = collection_name
        self._transport_factory = transport_factory
        self._collection_opener = collection_opener

    def _transport(self) -> Any:
        if self._transport_factory is not None:
            return self._transport_factory(self.base_url)
        from actian_vectorai.transport import RESTTransport

        return RESTTransport(self.base_url)

    async def _call(self, method_name: str, *args: object, **kwargs: object) -> Any:
        transport = self._transport()
        try:
            return await getattr(transport, method_name)(*args, **kwargs)
        except CollectionSafetyError:
            raise
        except Exception as error:
            raise MemoryBackendError(f"VectorAI REST {method_name} failed") from error
        finally:
            close = getattr(transport, "close", None)
            if close is not None:
                await close()

    async def health_check(self) -> dict[str, Any]:
        result = await self._call("health_check", timeout=10.0)
        if not isinstance(result, dict):
            raise MemoryBackendError("VectorAI health response was malformed")
        return result

    async def list_collections(self) -> list[str]:
        result = await self._call("collections_list", timeout=10.0)
        if not isinstance(result, list) or not all(isinstance(name, str) for name in result):
            raise MemoryBackendError("VectorAI collection list was malformed")
        return sorted(result)

    async def ensure_collection(self) -> str:
        """List first, then create only the one allowed collection when safe."""

        names = await self.list_collections()
        unexpected = [name for name in names if name != self.collection_name]
        if unexpected:
            raise CollectionSafetyError(
                "isolated database contains unexpected collections: " + ", ".join(unexpected)
            )
        if self.collection_name in names:
            info = await self._call("collections_get", self.collection_name, timeout=10.0)
            self._validate_collection_schema(info)
            await self._open_if_needed(info)
            return "existing"
        created = await self._call(
            "collections_create",
            self.collection_name,
            {"vectors": {"size": EMBEDDING_DIMENSION, "distance": "Cosine"}},
            timeout=30.0,
        )
        if created is not True:
            raise MemoryBackendError("VectorAI did not confirm collection creation")
        info = await self._call("collections_get", self.collection_name, timeout=10.0)
        self._validate_collection_schema(info)
        return "created"

    async def open_existing_collection(self, *, require_empty: bool = False) -> dict[str, Any]:
        """Open only the sole expected collection; never create or replace one."""

        names = await self.list_collections()
        if names != [self.collection_name]:
            raise CollectionSafetyError(
                "expected only the existing project collection; found " + repr(names)
            )
        info = await self.get_collection_schema()
        if require_empty and info.get("points_count") != 0:
            raise CollectionSafetyError("project collection must be empty before the benchmark")
        await self._open_if_needed(info)
        return info

    async def _open_if_needed(self, info: Mapping[str, Any]) -> None:
        if not self._schema_reports_open(info):
            opened = await self._open_collection()
            if opened is not True:
                raise MemoryBackendError("VectorAI did not confirm collection open")

    @staticmethod
    def _schema_reports_open(info: object) -> bool:
        if not isinstance(info, Mapping):
            return False
        return (
            str(info.get("status", "")).casefold() == "green"
            and str(info.get("health_status_ext", "")).casefold() == "health_green"
        )

    async def _open_collection(self) -> bool:
        if self._collection_opener is not None:
            try:
                return await self._collection_opener(self.grpc_url, self.collection_name)
            except Exception as error:
                raise MemoryBackendError("VectorAI collection open failed") from error
        try:
            from actian_vectorai import AsyncVectorAIClient

            async with AsyncVectorAIClient(url=self.grpc_url) as client:
                return await client.vde.open_collection(self.collection_name, timeout=30.0)
        except Exception as error:
            raise MemoryBackendError("VectorAI collection open failed") from error

    async def get_collection_schema(self) -> dict[str, Any]:
        """Return the server schema for evidence without mutating the collection."""

        info = await self._call("collections_get", self.collection_name, timeout=10.0)
        self._validate_collection_schema(info)
        return dict(info)

    @staticmethod
    def _validate_collection_schema(info: object) -> None:
        if not isinstance(info, Mapping):
            raise CollectionSafetyError("collection schema response is malformed")

        def find_schema(value: object) -> tuple[object, object] | None:
            if isinstance(value, Mapping):
                if "size" in value and "distance" in value:
                    return value["size"], value["distance"]
                for nested in value.values():
                    found = find_schema(nested)
                    if found is not None:
                        return found
            return None

        schema = find_schema(info)
        if schema is None:
            raise CollectionSafetyError("collection schema lacks vector size and distance")
        size, distance = schema
        if size != EMBEDDING_DIMENSION or str(distance).casefold() != "cosine":
            raise CollectionSafetyError(
                f"collection schema mismatch: size={size!r}, distance={distance!r}"
            )

    async def upsert(self, record: MemoryRecord) -> None:
        if not isinstance(record, MemoryRecord):
            raise MemoryValidationError("record must be a MemoryRecord")
        result = await self._call(
            "points_upsert",
            self.collection_name,
            [{"id": record.memory_id, "vector": list(record.embedding), "payload": record.payload}],
            timeout=30.0,
        )
        if not isinstance(result, Mapping):
            raise MemoryBackendError("VectorAI upsert response was malformed")

    async def search(
        self, embedding: tuple[float, ...], *, top_k: int, error_type: str | None
    ) -> list[MemorySearchResult]:
        vector = _validated_embedding(embedding)
        _validate_search_arguments(top_k, error_type)
        metadata_filter = None
        if error_type is not None:
            metadata_filter = {"must": [{"key": "error_type", "match": {"value": error_type}}]}
        hits = await self._call(
            "points_search",
            self.collection_name,
            list(vector),
            top_k,
            filter=metadata_filter,
            with_payload=True,
            with_vector=True,
            timeout=30.0,
        )
        if not isinstance(hits, list):
            raise MemoryBackendError("VectorAI search response was malformed")
        try:
            return [self._parse_hit(hit) for hit in hits]
        except (MemoryValidationError, KeyError, TypeError) as error:
            raise MemoryBackendError("VectorAI returned a malformed search record") from error

    @staticmethod
    def _parse_hit(hit: object) -> MemorySearchResult:
        if not isinstance(hit, Mapping):
            raise MemoryValidationError("search hit must be an object")
        embedding = hit.get("vector", hit.get("vectors"))
        if isinstance(embedding, Mapping):
            embedding = embedding.get("")
        return MemorySearchResult(
            record=MemoryRecord.from_storage(
                hit.get("payload"), embedding, fallback_id=hit.get("id")
            ),
            score=hit["score"],  # type: ignore[arg-type]
        )

    async def delete(self, memory_id: str) -> None:
        result = await self._call(
            "points_delete",
            self.collection_name,
            ids=[_required_text(memory_id, "memory_id")],
            timeout=30.0,
        )
        if not isinstance(result, Mapping):
            raise MemoryBackendError("VectorAI delete response was malformed")


class MemoryService:
    """Tool-facing service that stores only outcomes confirmed by the current run."""

    def __init__(self, backend: MemoryBackend, embedder: Embedder, *, session_id: str) -> None:
        self.backend = backend
        self.embedder = embedder
        self.session_id = _required_text(session_id, "session_id")
        self._confirmed_outcomes: set[str] = set()

    def confirm_outcome(self, outcome: str) -> None:
        self._confirmed_outcomes.add(_required_text(outcome, "outcome"))

    async def search_codebase_memory(
        self, query: str, *, error_type: str | None = None, top_k: int = 3
    ) -> str:
        clean_query = _required_text(query, "query")
        _validate_search_arguments(top_k, error_type)
        embedding = await self.embedder.embed(clean_query)
        results = await self.backend.search(embedding, top_k=top_k, error_type=error_type)
        return format_search_results(results)

    async def store_fix_memory(
        self,
        *,
        fix_description: str,
        error_type: str,
        file_paths: Sequence[str],
        outcome: str,
        timestamp: datetime | None = None,
    ) -> str:
        clean_fix = _required_text(fix_description, "fix_description")
        clean_error = _required_text(error_type, "error_type")
        clean_outcome = _required_text(outcome, "outcome")
        if clean_outcome not in self._confirmed_outcomes:
            raise UnconfirmedOutcomeError("outcome was not confirmed by the current run")
        if isinstance(file_paths, (str, bytes)):
            raise MemoryValidationError("file_paths must be a sequence of paths")
        paths = tuple(file_paths)
        confirmed_at = timestamp or datetime.now(UTC)
        if confirmed_at.tzinfo is None or confirmed_at.utcoffset() is None:
            raise MemoryValidationError("timestamp must include a timezone")
        memory_id = str(
            uuid5(
                NAMESPACE_URL,
                "\n".join((self.session_id, clean_fix, clean_error, clean_outcome, *paths)),
            )
        )
        record = MemoryRecord(
            memory_id=memory_id,
            fix_description=clean_fix,
            error_type=clean_error,
            outcome=clean_outcome,
            file_paths=paths,
            timestamp=confirmed_at.isoformat(),
            session_id=self.session_id,
            embedding=await self.embedder.embed(clean_fix),
        )
        await self.backend.upsert(record)
        return memory_id


def format_search_results(results: Sequence[MemorySearchResult]) -> str:
    """Return compact readable evidence for a function-call output."""

    if not results:
        return "No relevant confirmed fixes found."
    blocks = []
    for index, result in enumerate(results, start=1):
        record = result.record
        blocks.append(
            "\n".join(
                (
                    f"Match {index} (score={result.score:.4f})",
                    f"Fix: {record.fix_description}",
                    f"Error type: {record.error_type}",
                    f"Files: {', '.join(record.file_paths)}",
                    f"Outcome: {record.outcome}",
                    f"Confirmed: {record.timestamp}",
                    f"Session: {record.session_id}",
                )
            )
        )
    return "\n\n".join(blocks)


async def search_codebase_memory(
    service: MemoryService, query: str, error_type: str | None = None, top_k: int = 3
) -> str:
    """Asynchronous tool boundary for local semantic retrieval."""

    return await service.search_codebase_memory(query, error_type=error_type, top_k=top_k)


async def store_fix_memory(
    service: MemoryService,
    *,
    fix_description: str,
    error_type: str,
    file_paths: Sequence[str],
    outcome: str,
) -> str:
    """Tool boundary for storing one current-run confirmed fix."""

    return await service.store_fix_memory(
        fix_description=fix_description,
        error_type=error_type,
        file_paths=file_paths,
        outcome=outcome,
    )


MEMORY_TOOL_DEFINITIONS = (
    {
        "type": "function",
        "name": "search_codebase_memory",
        "description": "Search confirmed past debugging results using observed failure symptoms.",
        "async": True,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "error_type": {"type": ["string", "null"]},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query", "error_type", "top_k"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "store_fix_memory",
        "description": (
            "Store a debugging result after its outcome is confirmed by the current run."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "fix_description": {"type": "string"},
                "error_type": {"type": "string"},
                "file_paths": {"type": "array", "items": {"type": "string"}},
                "outcome": {"type": "string"},
            },
            "required": ["fix_description", "error_type", "file_paths", "outcome"],
            "additionalProperties": False,
        },
    },
)
