"""Streaming, resumable, idempotent migration: Chroma -> Actian VectorAI DB.

Safety model
------------
* Upserts use deterministic point IDs, so re-sending a batch is harmless (idempotent).
* A checkpoint (state.json) records the next Chroma offset after every fully-written page, plus a
  fingerprint of everything that influences the output. ``--resume`` refuses to continue if it differs.
* Offset paging is only consistent on a frozen collection, so the source count is re-read at the end
  and the run FAILS if it changed (writes were not paused).
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from actian_vectorai import Distance

from . import __version__
from .errors import Interrupted, MigrationError, UsageError
from .progress import Progress
from .source import ChromaSource, Page
from .state import Checkpoint, DeadLetter, write_json
from .target import ActianTarget
from .transform import MAX_DIMENSION, RecordTransformer, TransformOptions, chunk_points, resolve_distance

log = logging.getLogger(__name__)

IF_EXISTS = ("fail", "reuse", "recreate")
_SENTINEL = object()


@dataclass
class MigrationOptions:
    collection: str
    target_collection: str
    transform: TransformOptions
    workdir: Path = Path("migration_work")
    batch_size: int = 256
    max_batch_bytes: int = 16 * 1024 * 1024
    workers: int = 4
    if_exists: str = "fail"
    assume_yes: bool = False
    resume: bool = False
    fresh: bool = False
    on_bad_record: str = "fail"          # fail | skip
    allow_zero_vectors: bool = False
    distance_override: str | None = None
    dimension_override: int | None = None
    hnsw_m: int | None = None
    hnsw_ef_construct: int | None = None
    flush: bool = True
    progress_interval: float = 10.0
    stop_event: threading.Event = field(default_factory=threading.Event)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _prefetch(it: Iterator[Page]) -> Iterator[Page]:
    """Read page k+1 from Chroma while page k is being written to Actian."""
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chroma-read")
    try:
        fut = ex.submit(next, it, _SENTINEL)
        while True:
            item = fut.result()
            if item is _SENTINEL:
                return
            fut = ex.submit(next, it, _SENTINEL)
            yield item  # type: ignore[misc]
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def migrate_collection(source: ChromaSource, target: ActianTarget, opts: MigrationOptions) -> dict[str, Any]:
    t_start = time.monotonic()
    if opts.if_exists not in IF_EXISTS:
        raise UsageError(f"--if-exists must be one of {IF_EXISTS}")
    if opts.on_bad_record not in ("fail", "skip"):
        raise UsageError("--on-bad-record must be fail or skip")
    if opts.batch_size < 1 or opts.workers < 1:
        raise UsageError("--batch-size and --workers must be >= 1")
    if opts.resume and opts.fresh:
        raise UsageError("--resume and --fresh are mutually exclusive")

    ckpt = Checkpoint(opts.workdir)
    dead = DeadLetter(opts.workdir)
    coll = source.get_collection(opts.collection)
    n0 = source.count(coll)
    space, space_src = source.space(coll)
    distance = resolve_distance(space, opts.distance_override)
    log.info("Source %r: %d records, Chroma space=%s (%s) -> Actian %s", opts.collection, n0, space, space_src, distance.name)

    if n0 == 0 and opts.dimension_override is None:
        log.warning("Source collection is empty: nothing to migrate (pass --dimension N to create an empty target).")
        summary = {"collection": opts.collection, "status": "empty", "source_count": 0, "migrated": 0}
        write_json(opts.workdir / "migration_summary.json", summary)
        return summary

    # ---- dimension from the first record (every later record is checked against it)
    dimension = opts.dimension_override
    if dimension is None:
        first = source.fetch(coll, 0, min(source.page_size, 16), ("embeddings",))
        if not first.ids or first.embeddings is None:
            raise MigrationError("could not read any embedding from the source collection")
        dimension = int(len(first.embeddings[0]))
    if dimension > MAX_DIMENSION:
        raise UsageError(f"dimension {dimension} exceeds Actian's maximum of {MAX_DIMENSION}")

    topts = opts.transform
    topts.reject_zero_vectors = distance == Distance.Cosine and not opts.allow_zero_vectors
    transformer = RecordTransformer(topts, dimension=dimension)

    # ---- checkpoint / resume decision
    fingerprint = {
        "source": source.describe(), "collection": opts.collection,
        "target": target.cfg.describe(), "target_collection": opts.target_collection,
        "distance": distance.name, "dimension": dimension, "on_bad_record": opts.on_bad_record,
        **topts.fingerprint(),
    }
    if opts.fresh:
        ckpt.clear()
    state = ckpt.load()
    start_offset = 0
    migrated = skipped = 0
    preexisting = 0
    if state is not None:
        if state.get("fingerprint") != fingerprint:
            raise UsageError(f"{ckpt.path} belongs to a different migration (settings changed). "
                             "Use --fresh to discard it and start over.")
        if state.get("completed"):
            raise UsageError(f"{ckpt.path}: this migration already completed. Use --fresh to run it again.")
        if not opts.resume:
            raise UsageError(f"found an unfinished migration at offset {state.get('next_offset')} in {ckpt.path}. "
                             "Pass --resume to continue it or --fresh to start over.")
        if state.get("source_count") != n0:
            raise UsageError(f"source count changed since the run started ({state.get('source_count')} -> {n0}): "
                             "writes were not frozen, offsets are no longer reliable. Use --fresh.")
        start_offset = int(state["next_offset"])
        migrated, skipped = int(state.get("migrated", 0)), int(state.get("skipped", 0))
        preexisting = int(state.get("preexisting_points", 0))
        log.info("Resuming at offset %d (%d migrated, %d skipped so far)", start_offset, migrated, skipped)
    elif opts.resume:
        log.warning("--resume given but no checkpoint exists; starting from the beginning")

    # ---- target collection
    if state is None:
        preexisting = _prepare_target(target, opts, dimension, distance)
    else:
        _verify_target(target, opts.target_collection, dimension, distance)

    state = state or {}
    state.update({
        "version": 1, "tool_version": __version__, "fingerprint": fingerprint, "source_count": n0,
        "next_offset": start_offset, "migrated": migrated, "skipped": skipped, "preexisting_points": preexisting,
        "completed": False, "started_at": state.get("started_at") or _now(),
    })
    ckpt.save(state)

    # ---- stream
    prog = Progress(n0, label="migrated", interval=opts.progress_interval, done=start_offset)
    pages = _prefetch(source.iter_pages(coll, start_offset=start_offset))
    pool = ThreadPoolExecutor(max_workers=opts.workers, thread_name_prefix="actian-write") if opts.workers > 1 else None
    try:
        for page in pages:
            converted = transformer.convert_page(page)
            if converted.rejects:
                if opts.on_bad_record == "fail":
                    r = converted.rejects[0]
                    raise MigrationError(
                        f"record {r.chroma_id!r} (offset {r.offset}) cannot be converted: {r.reason} "
                        f"({len(converted.rejects)} bad record(s) in this page). Fix the source or use "
                        "--on-bad-record skip. Run `preflight` for the full list.")
                dead.append(converted.rejects)
            batches = list(chunk_points(converted.points, converted.sizes, opts.batch_size, opts.max_batch_bytes))
            if pool is not None and len(batches) > 1:
                for _ in pool.map(lambda b: target.upsert(opts.target_collection, b), batches):
                    pass
            else:
                for b in batches:
                    target.upsert(opts.target_collection, b)
            migrated += len(converted.points)
            skipped += len(converted.rejects)
            state.update({"next_offset": page.next_offset, "migrated": migrated, "skipped": skipped,
                          "updated_at": _now()})
            ckpt.save(state)
            prog.advance(len(page.ids))
            if opts.stop_event.is_set():
                raise Interrupted(f"stopped at offset {page.next_offset}; checkpoint saved. Re-run with --resume.")
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        pages.close()
    prog.advance(0, force=True)

    # ---- post-conditions
    n1 = source.count(coll)
    if n1 != n0:
        state["source_changed"] = {"before": n0, "after": n1}
        ckpt.save(state)
        raise MigrationError(f"source count changed during the migration ({n0} -> {n1}): writes were not paused, "
                             "so the target is not a consistent snapshot. Pause writes and re-run with --fresh.")
    if state["next_offset"] < n0:
        raise MigrationError(f"scan ended at offset {state['next_offset']} but the source has {n0} records")

    if opts.flush:
        try:
            target.flush(opts.target_collection)
            log.info("Collection state: %s", target.wait_ready(opts.target_collection))
        except MigrationError as exc:
            log.warning("flush/ready check failed (data was written): %s", exc)

    target_count = target.count(opts.target_collection)
    expected = n0 - skipped
    exact = preexisting == 0
    count_ok = (target_count == expected) if exact else (target_count >= expected)
    state.update({"completed": count_ok, "finished_at": _now(), "target_count": target_count})
    ckpt.save(state)

    summary = {
        "collection": opts.collection, "target_collection": opts.target_collection,
        "status": "ok" if count_ok else "count_mismatch", "source_count": n0, "migrated": migrated,
        "skipped": skipped, "target_count": target_count, "expected_target_count": expected,
        "preexisting_points": preexisting, "dimension": dimension, "distance": distance.name,
        "chroma_space": space, "id_strategy": topts.id_strategy.value, "uuid_namespace": str(topts.uuid_namespace),
        "id_field": topts.id_field, "document_field": topts.document_field,
        "metadata_key_renames": transformer.field_map, "transform_counters": dict(transformer.counters),
        "duration_seconds": round(time.monotonic() - t_start, 2), "finished_at": _now(), "tool_version": __version__,
    }
    write_json(opts.workdir / "migration_summary.json", summary)
    if not count_ok:
        raise MigrationError(f"target count {target_count} does not match expected {expected} "
                             f"({n0} source - {skipped} skipped). Search may lag writes: re-run `validate`.")
    log.info("Migrated %d records (%d skipped) into %r in %.1fs", migrated, skipped, opts.target_collection,
             summary["duration_seconds"])
    return summary


def _verify_target(target: ActianTarget, name: str, dimension: int, distance: Distance) -> None:
    if not target.exists(name):
        raise UsageError(f"cannot resume: target collection {name!r} no longer exists. Use --fresh.")
    info = target.info(name)
    if info.named_vectors:
        raise UsageError(f"target collection {name!r} uses named vectors, which this tool does not write")
    if info.dimension != dimension or info.distance != distance:
        raise UsageError(f"target collection {name!r} is dimension={info.dimension}, distance="
                         f"{info.distance.name if info.distance else None}; the migration needs "
                         f"{dimension}/{distance.name}")


def _prepare_target(target: ActianTarget, opts: MigrationOptions, dimension: int, distance: Distance) -> int:
    """Create or validate the destination; returns the number of points already in it."""
    name = opts.target_collection
    if target.exists(name):
        if opts.if_exists == "recreate":
            if not opts.assume_yes:
                raise UsageError(f"--if-exists recreate would DROP collection {name!r}. Add --yes to confirm.")
            target.drop(name)
        else:
            info = target.info(name)
            if opts.if_exists == "fail" and info.points > 0:
                raise UsageError(
                    f"target collection {name!r} already exists with {info.points} points. "
                    "Use --if-exists reuse (merge/upsert) or recreate (drop first, needs --yes).")
            _verify_target(target, name, dimension, distance)
            log.info("Reusing existing collection %r (%d points)", name, info.points)
            return info.points
    target.create(name, dimension, distance, hnsw_m=opts.hnsw_m, hnsw_ef_construct=opts.hnsw_ef_construct)
    return 0
