"""Chroma reader: local persistent, self-hosted HTTP server, or Chroma Cloud (same Collection API)."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from .config import ChromaConfig
from .errors import SourceError, UsageError
from .retry import backoff_delay, retry_call

log = logging.getLogger(__name__)

MIN_PAGE_SIZE = 10
DEFAULT_INCLUDE = ("embeddings", "documents", "metadatas")


def _error_classes(*names: str) -> tuple[type[BaseException], ...]:
    try:
        from chromadb import errors as chroma_errors
    except Exception:  # pragma: no cover
        return ()
    return tuple(getattr(chroma_errors, n) for n in names if hasattr(chroma_errors, n))


# Deterministic failures: retrying cannot help.
_FATAL = _error_classes("ChromaAuthError", "AuthorizationError", "NotFoundError", "InvalidArgumentError",
                        "InvalidDimensionException") + (ValueError, TypeError, KeyError)
# Failures that usually mean "that page was too large": shrink the page, then retry.
_SHRINK = _error_classes("BatchSizeExceededError", "QuotaError")


@dataclass
class Page:
    offset: int                       # position of ids[0] in the collection's scan order
    ids: list[str]
    embeddings: Any = None            # ndarray (n, d) or sequence of vectors; None when not requested
    documents: list[str | None] | None = None
    metadatas: list[dict | None] | None = None
    uris: list[str | None] | None = None
    next_offset: int = 0


class ChromaSource:
    def __init__(self, cfg: ChromaConfig, *, client: Any | None = None, max_attempts: int = 6,
                 sleep: Any = None) -> None:
        self.cfg = cfg
        self.page_size = cfg.page_size
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._client = client

    # ------------------------------------------------------------ connection
    def connect(self) -> None:
        if self._client is not None:
            return
        self.cfg.validate()
        import chromadb

        c = self.cfg
        try:
            if c.mode == "local":
                self._client = chromadb.PersistentClient(path=c.path, **self._tenant_kwargs())
            elif c.mode == "http":
                headers = dict(c.headers)
                if c.token:
                    headers.setdefault("Authorization", f"Bearer {c.token}")
                self._client = chromadb.HttpClient(host=c.host, port=c.port, ssl=c.ssl,
                                                   headers=headers or None, **self._tenant_kwargs())
            else:
                self._client = chromadb.CloudClient(tenant=c.tenant, database=c.database, api_key=c.api_key,
                                                    cloud_host=c.cloud_host)
        except Exception as exc:  # noqa: BLE001
            raise UsageError(f"cannot open Chroma source ({c.describe()}): {type(exc).__name__}: {exc}") from exc

    def _tenant_kwargs(self) -> dict[str, str]:
        kw: dict[str, str] = {}
        if self.cfg.tenant:
            kw["tenant"] = self.cfg.tenant
        if self.cfg.database:
            kw["database"] = self.cfg.database
        return kw

    @property
    def client(self) -> Any:
        if self._client is None:
            self.connect()
        return self._client

    def describe(self) -> str:
        return self.cfg.describe()

    # ------------------------------------------------------------ discovery
    def list_collections(self) -> list[str]:
        names: list[str] = []
        offset, step = 0, 100
        while True:
            batch = self._call(lambda o=offset: self.client.list_collections(limit=step, offset=o), "list collections")
            batch = list(batch)
            names.extend(c if isinstance(c, str) else c.name for c in batch)
            if len(batch) < step:
                return names
            offset += step

    def get_collection(self, name: str) -> Any:
        try:
            return self._call(lambda: self.client.get_collection(name), f"open collection {name!r}")
        except SourceError as exc:
            raise UsageError(f"Chroma collection {name!r} cannot be opened: {exc}") from exc

    def count(self, coll: Any) -> int:
        return int(self._call(coll.count, "count"))

    def space(self, coll: Any) -> tuple[str, str]:
        """Return (space, where_it_came_from); space is one of l2 | cosine | ip."""
        cfg = getattr(coll, "configuration", None)
        for key in ("hnsw", "spann"):
            sub = cfg.get(key) if isinstance(cfg, Mapping) else getattr(cfg, key, None)
            if sub:
                s = sub.get("space") if isinstance(sub, Mapping) else getattr(sub, "space", None)
                if s:
                    return str(s), "configuration"
        meta = getattr(coll, "metadata", None) or {}
        if meta.get("hnsw:space"):
            return str(meta["hnsw:space"]), "metadata"
        return "l2", "default"

    # --------------------------------------------------------------- reading
    def _call(self, fn: Any, what: str) -> Any:
        kwargs: dict[str, Any] = {}
        if self._sleep is not None:
            kwargs["sleep"] = self._sleep
        try:
            return retry_call(fn, attempts=self.max_attempts, what=f"Chroma {what}",
                              is_retryable=lambda e: not isinstance(e, _FATAL), **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"Chroma {what} failed: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _page_from(raw: Mapping[str, Any], offset: int) -> Page:
        ids = list(raw.get("ids") or [])
        return Page(offset=offset, ids=ids, embeddings=raw.get("embeddings"), documents=raw.get("documents"),
                    metadatas=raw.get("metadatas"), uris=raw.get("uris"), next_offset=offset + len(ids))

    def fetch(self, coll: Any, offset: int, limit: int, include: Sequence[str] = DEFAULT_INCLUDE) -> Page:
        """One page. May return fewer than ``limit`` records if the page size had to shrink: never infer
        end-of-data from a short page; an empty page is the only terminator."""
        size = limit
        failures = 0
        while True:
            try:
                raw = coll.get(include=list(include), limit=size, offset=offset)
                return self._page_from(raw, offset)
            except _FATAL as exc:
                raise SourceError(f"Chroma get(offset={offset}, limit={size}) failed: {type(exc).__name__}: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                failures += 1
                too_big = isinstance(exc, _SHRINK) or failures >= 2
                if too_big and size > MIN_PAGE_SIZE:
                    size = max(MIN_PAGE_SIZE, size // 2)
                    self.page_size = min(self.page_size, size)
                    log.warning("Chroma get failed (%s); shrinking page size to %d", type(exc).__name__, size)
                    continue
                if failures >= self.max_attempts:
                    raise SourceError(f"Chroma get(offset={offset}) failed after {failures} attempts: "
                                      f"{type(exc).__name__}: {exc}") from exc
                (self._sleep or time.sleep)(backoff_delay(failures))

    def iter_pages(self, coll: Any, *, start_offset: int = 0, include: Sequence[str] = DEFAULT_INCLUDE) -> Iterator[Page]:
        offset = start_offset
        while True:
            page = self.fetch(coll, offset, self.page_size, include)
            if not page.ids:
                return
            offset = page.next_offset
            yield page

    def get_by_ids(self, coll: Any, ids: Sequence[str], include: Sequence[str] = DEFAULT_INCLUDE) -> Page:
        raw = self._call(lambda: coll.get(ids=list(ids), include=list(include)), "get by ids")
        return self._page_from(raw, 0)

    def query(self, coll: Any, vector: Sequence[float], k: int) -> list[str]:
        raw = self._call(lambda: coll.query(query_embeddings=[list(map(float, vector))], n_results=k, include=[]),
                         "query")
        return list((raw.get("ids") or [[]])[0])
