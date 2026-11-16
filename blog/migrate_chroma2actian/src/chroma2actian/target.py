"""Actian VectorAI DB writer/reader built on the official Python SDK (`actian-vectorai-client`)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from actian_vectorai import (
    CapacityExceededError,
    Distance,
    HnswConfigDiff,
    MessageSizeExceededError,
    PointStruct,
    UpdateStatus,
    VectorAIClient,
    VectorAIError,
    VectorParams,
    get_retry_delay,
    is_retryable,
)

from .config import ActianConfig
from .errors import CapacityError, TargetError, UsageError
from .retry import retry_call

log = logging.getLogger(__name__)


@dataclass
class TargetCollectionInfo:
    dimension: int | None
    distance: Distance | None
    points: int
    named_vectors: bool = False


def build_client(cfg: ActianConfig) -> Any:
    return VectorAIClient(cfg.url, **cfg.client_kwargs()) if cfg.url else VectorAIClient(**cfg.client_kwargs())


class ActianTarget:
    def __init__(self, cfg: ActianConfig, *, client: Any | None = None, max_attempts: int = 5, sleep: Any = None) -> None:
        self.cfg = cfg
        self.client = client if client is not None else build_client(cfg)
        self.max_attempts = max_attempts
        self._sleep = sleep or time.sleep
        self._connected = False

    # ------------------------------------------------------------ lifecycle
    def connect(self) -> dict[str, Any]:
        """Connect and verify health, optionally waiting for a container that is still starting."""
        deadline = time.monotonic() + max(0.0, self.cfg.wait_for_target)
        last: Exception | None = None
        while True:
            try:
                if not self._connected:
                    self.client.connect()
                    self._connected = True
                info = self.client.health_check()
                log.info("Connected to %s v%s at %s", info.get("title"), info.get("version"), self.cfg.describe())
                return info
            except Exception as exc:  # noqa: BLE001
                last = exc
                self._connected = False
                if time.monotonic() >= deadline:
                    break
                self._sleep(2.0)
        raise UsageError(
            f"cannot reach Actian VectorAI DB at {self.cfg.describe()}: {type(last).__name__}: {last}\n"
            "  - is the container running?  docker ps   (gRPC is 6574, REST 6573)\n"
            "  - non-local host with a token needs --actian-tls (or --actian-allow-insecure)\n"
            "  - use --wait-for-target SECONDS if the container is still starting"
        ) from last

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            log.debug("error while closing Actian client", exc_info=True)

    # -------------------------------------------------------------- helpers
    def _call(self, fn: Any, what: str) -> Any:
        try:
            return retry_call(fn, attempts=self.max_attempts, what=f"Actian {what}", sleep=self._sleep,
                              is_retryable=lambda e: isinstance(e, VectorAIError) and is_retryable(e),
                              delay_for=lambda e, n: get_retry_delay(e, n))
        except CapacityExceededError as exc:
            raise CapacityError(
                f"Actian licence capacity exhausted during {what}: {exc}\n"
                "  Free capacity (delete vectors), upgrade the licence, or migrate fewer vectors.") from exc
        except VectorAIError as exc:
            raise TargetError(f"Actian {what} failed: {type(exc).__name__}: {exc}") from exc

    # ---------------------------------------------------------- collections
    def exists(self, name: str) -> bool:
        return bool(self._call(lambda: self.client.collections.exists(name), f"exists({name})"))

    def info(self, name: str) -> TargetCollectionInfo:
        ci = self._call(lambda: self.client.collections.get_info(name), f"get_info({name})")
        params = ci.config.params if ci.config is not None else None
        vc = params.vectors_config if params is not None else None
        named = bool(params is not None and params.vectors_config_map)
        points = ci.points_count if ci.points_count is not None else (ci.vectors_count or 0)
        return TargetCollectionInfo(dimension=vc.size if vc else None, distance=vc.distance if vc else None,
                                    points=int(points), named_vectors=named)

    def create(self, name: str, dimension: int, distance: Distance, *, hnsw_m: int | None = None,
               hnsw_ef_construct: int | None = None) -> None:
        kwargs: dict[str, Any] = {"vectors_config": VectorParams(size=dimension, distance=distance)}
        if hnsw_m is not None or hnsw_ef_construct is not None:
            hk = {k: v for k, v in (("m", hnsw_m), ("ef_construct", hnsw_ef_construct)) if v is not None}
            kwargs["hnsw_config"] = HnswConfigDiff(**hk)
        self._call(lambda: self.client.collections.create(name, **kwargs), f"create({name})")
        log.info("Created collection %r (dimension=%d, distance=%s)", name, dimension, distance.name)

    def drop(self, name: str) -> None:
        self._call(lambda: self.client.collections.delete(name), f"delete({name})")
        log.warning("Dropped collection %r", name)

    def count(self, name: str) -> int:
        return int(self._call(lambda: self.client.points.count(name), f"count({name})"))

    def flush(self, name: str) -> None:
        self._call(lambda: self.client.vde.flush(name), f"flush({name})")

    def wait_ready(self, name: str, timeout: float = 300.0) -> str:
        state = self._call(lambda: self.client.vde.wait_for_ready(name, timeout=timeout), f"wait_for_ready({name})")
        return getattr(state, "name", str(state))

    # -------------------------------------------------------------- writing
    def upsert(self, name: str, points: Sequence[PointStruct]) -> int:
        """Upsert with retries; a message-size error splits the batch in half and recurses."""
        if not points:
            return 0
        try:
            res = self._call(lambda: self.client.points.upsert(name, list(points)), f"upsert({len(points)} points)")
        except TargetError as exc:
            if isinstance(exc.__cause__, MessageSizeExceededError) and len(points) > 1:
                mid = len(points) // 2
                log.warning("Batch of %d points exceeds the message limit; splitting", len(points))
                return self.upsert(name, points[:mid]) + self.upsert(name, points[mid:])
            raise
        status = getattr(res, "status", None)
        if status == UpdateStatus.ClockRejected:
            raise TargetError(f"Actian rejected an upsert batch of {len(points)} points (status=clock_rejected)")
        return len(points)

    # -------------------------------------------------------------- reading
    def scroll_ids(self, name: str, page_size: int = 1000) -> Iterator[list[Any]]:
        """Yield pages of point IDs (ids only; no payloads or vectors)."""
        offset = None
        while True:
            pts, nxt = self._call(
                lambda o=offset: self.client.points.scroll(name, offset=o, limit=page_size, with_payload=False,
                                                           with_vectors=False), f"scroll({name})")
            if not pts:
                return
            yield [p.id for p in pts]
            if nxt is None:
                return
            if nxt == offset:
                raise TargetError(f"scroll cursor did not advance past {offset!r}; aborting to avoid an endless loop")
            offset = nxt

    def get(self, name: str, ids: Sequence[int | str], *, with_payload: bool = True, with_vectors: bool = False) -> list[Any]:
        return list(self._call(lambda: self.client.points.get(name, ids=list(ids), with_payload=with_payload,
                                                              with_vectors=with_vectors), f"get({len(ids)} ids)"))

    def search(self, name: str, vector: Sequence[float], limit: int) -> list[Any]:
        return list(self._call(lambda: self.client.points.search(name, vector=list(vector), limit=limit),
                               "search"))
