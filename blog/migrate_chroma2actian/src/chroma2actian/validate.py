"""Post-migration verification. Every check is independent; the run passes only if all enabled checks pass.

Checks
------
config          target dimension + distance match the source
target_count    points.count == expected (source count minus records the migration skipped)
id_set          the set of IDs is identical, via an O(1)-memory order-independent digest (see IdSetDigest)
sample_records  a deterministic hash-sample of records is compared field by field (vector, payload)
search_parity   top-k overlap between Chroma query() and Actian search() on golden or self queries
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from actian_vectorai import Distance

from .errors import MigrationError, UsageError
from .source import ChromaSource
from .state import DeadLetter, read_json, write_json
from .target import ActianTarget
from .transform import (BadRecord, IdSetDigest, RecordTransformer, TransformOptions, canonical_id, map_id,
                        resolve_distance, sample_bucket)

log = logging.getLogger(__name__)


@dataclass
class ValidateOptions:
    collection: str
    target_collection: str
    transform: TransformOptions
    workdir: Path = Path("migration_work")
    distance_override: str | None = None
    sample: int = 500
    top_k: int = 10
    auto_queries: int = 20
    golden_queries: Path | None = None
    min_mean_overlap: float = 0.9
    min_query_overlap: float = 0.6
    poll_timeout: float = 300.0
    allow_extra_target_points: bool = False
    diagnose: bool = True
    diagnose_limit: int = 2_000_000
    report: Path | None = None
    allow_zero_vectors: bool = False
    progress_interval: float = 10.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: dict[str, Any] = field(default_factory=dict)


def _poll(fn: Any, ok: Any, timeout: float, interval: float = 2.0) -> Any:
    """Actian acknowledges writes before they are searchable: poll until ok(fn()) or timeout."""
    deadline = time.monotonic() + timeout
    while True:
        value = fn()
        if ok(value) or time.monotonic() >= deadline:
            return value
        time.sleep(interval)


def _norm_payload(v: Any) -> Any:
    if isinstance(v, float) and v != v:
        return "__nan__"
    if isinstance(v, dict):
        return {str(k): _norm_payload(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_norm_payload(x) for x in v]
    return v


def _vectors_match(expected: np.ndarray, actual: Any, distance: Distance) -> bool:
    a = np.asarray(actual, dtype=np.float64)
    e = expected.astype(np.float64)
    if a.shape != e.shape:
        return False
    if distance == Distance.Cosine:
        # the server may store a normalised copy: compare direction, not magnitude
        na, ne = np.linalg.norm(a), np.linalg.norm(e)
        if na == 0 or ne == 0:
            return na == ne
        return float(np.dot(a, e) / (na * ne)) >= 1 - 1e-5
    return bool(np.allclose(a, e, rtol=1e-5, atol=1e-6))


def _scan_source_ids(source: ChromaSource, coll: Any, opts: ValidateOptions, skipped: set[str],
                     total: int) -> tuple[IdSetDigest, list[str], int, set[str] | None]:
    digest = IdSetDigest()
    sample: list[str] = []
    buckets = max(1, total // max(opts.sample, 1))
    keep: set[str] | None = set() if opts.diagnose and total <= opts.diagnose_limit else None
    n = 0
    for page in source.iter_pages(coll, include=()):
        for cid in page.ids:
            n += 1
            if cid in skipped:
                continue
            try:
                canon = canonical_id(map_id(cid, opts.transform.id_strategy, opts.transform.uuid_namespace))
            except BadRecord:
                continue  # unmappable under the chosen strategy: the migration rejected it too
            digest.add(canon)
            if keep is not None:
                keep.add(canon)
            if len(sample) < opts.sample and sample_bucket(cid) % buckets == 0:
                sample.append(cid)
    return digest, sample, n, keep


def _scan_target_ids(target: ActianTarget, name: str, keep: bool) -> tuple[IdSetDigest, set[str] | None]:
    digest = IdSetDigest()
    ids: set[str] | None = set() if keep else None
    for page in target.scroll_ids(name, 1000):
        for pid in page:
            c = canonical_id(pid)
            digest.add(c)
            if ids is not None:
                ids.add(c)
    return digest, ids


def run_validation(source: ChromaSource, target: ActianTarget, opts: ValidateOptions) -> dict[str, Any]:
    t0 = time.monotonic()
    coll = source.get_collection(opts.collection)
    total = source.count(coll)
    space, _ = source.space(coll)
    distance = resolve_distance(space, opts.distance_override)
    summary = read_json(opts.workdir / "migration_summary.json") or {}
    skipped_ids = DeadLetter(opts.workdir).ids()
    allow_extra = opts.allow_extra_target_points or int(summary.get("preexisting_points", 0)) > 0
    checks: list[Check] = []

    if not target.exists(opts.target_collection):
        raise UsageError(f"target collection {opts.target_collection!r} does not exist")
    info = target.info(opts.target_collection)

    # ---- config
    first = source.fetch(coll, 0, min(source.page_size, 16), ("embeddings",)) if total else None
    dim = int(len(first.embeddings[0])) if first is not None and first.ids else summary.get("dimension")
    checks.append(Check("config", info.dimension == dim and info.distance == distance, {
        "source_dimension": dim, "target_dimension": info.dimension, "expected_distance": distance.name,
        "target_distance": info.distance.name if info.distance else None}))

    # ---- source scan (ids only)
    log.info("Scanning %d source IDs ...", total)
    src_digest, sample_ids, scanned, src_keep = _scan_source_ids(source, coll, opts, skipped_ids, total)
    source_stable = scanned == total
    expected = src_digest.count
    checks.append(Check("source_stable", source_stable, {"count": total, "scanned": scanned,
                        "hint": None if source_stable else "the source changed while validating; freeze writes"}))

    # ---- counts + id set
    target_count = _poll(lambda: target.count(opts.target_collection),
                         (lambda c: c >= expected) if allow_extra else (lambda c: c == expected), opts.poll_timeout)
    count_ok = target_count >= expected if allow_extra else target_count == expected
    checks.append(Check("target_count", count_ok, {"expected": expected, "actual": target_count,
                        "skipped_by_migration": len(skipped_ids), "allow_extra": allow_extra}))

    log.info("Scanning Actian IDs ...")
    if allow_extra:
        missing = _presence_check(target, opts.target_collection, source, coll, opts, skipped_ids)
        checks.append(Check("id_set", not missing["missing"], {"mode": "subset (target may hold extra points)", **missing}))
    else:
        tgt_digest, tgt_ids = _scan_target_ids(target, opts.target_collection, src_keep is not None)
        ok = tgt_digest == src_digest
        detail: dict[str, Any] = {"mode": "exact", "source_digest": src_digest.hexdigest(),
                                  "target_digest": tgt_digest.hexdigest()}
        if not ok:
            if src_keep is not None and tgt_ids is not None:
                miss, extra = src_keep - tgt_ids, tgt_ids - src_keep
                detail.update({"missing_in_target": len(miss), "unexpected_in_target": len(extra),
                               "missing_examples": sorted(miss)[:10], "unexpected_examples": sorted(extra)[:10]})
            else:
                detail["note"] = "collection too large for --diagnose; counts differ or sets differ"
        checks.append(Check("id_set", ok, detail))

    # ---- sample records
    transformer = RecordTransformer(_zero_aware(opts, distance), dimension=dim)
    samples = _check_samples(source, target, coll, opts, transformer, sample_ids, distance)
    checks.append(samples["check"])

    # ---- search parity
    checks.append(_check_search(source, target, coll, opts, samples["vectors"], dim, total, expected))

    ok = all(c.ok for c in checks)
    report = {"ok": ok, "collection": opts.collection, "target_collection": opts.target_collection,
              "duration_seconds": round(time.monotonic() - t0, 2),
              "checks": [{"name": c.name, "ok": c.ok, **c.detail} for c in checks]}
    write_json(opts.report or (opts.workdir / "validation_report.json"), report)
    for c in checks:
        log.log(logging.INFO if c.ok else logging.ERROR, "[%s] %s %s", "PASS" if c.ok else "FAIL", c.name,
                json.dumps(c.detail, default=str)[:600])
    return report


def _zero_aware(opts: ValidateOptions, distance: Distance) -> TransformOptions:
    t = opts.transform
    t.reject_zero_vectors = distance == Distance.Cosine and not opts.allow_zero_vectors
    return t


def _presence_check(target: ActianTarget, name: str, source: ChromaSource, coll: Any, opts: ValidateOptions,
                    skipped: set[str]) -> dict[str, Any]:
    missing = 0
    examples: list[str] = []
    batch: list[tuple[int | str, str]] = []

    def flush() -> None:
        nonlocal missing
        if not batch:
            return
        found = {canonical_id(p.id) for p in target.get(name, [b[0] for b in batch], with_payload=False)}
        for pid, cid in batch:
            if canonical_id(pid) not in found:
                missing += 1
                if len(examples) < 10:
                    examples.append(cid)
        batch.clear()

    for page in source.iter_pages(coll, include=()):
        for cid in page.ids:
            if cid in skipped:
                continue
            try:
                batch.append((map_id(cid, opts.transform.id_strategy, opts.transform.uuid_namespace), cid))
            except BadRecord:
                continue
            if len(batch) >= 500:
                flush()
    flush()
    return {"missing": missing, "missing_examples": examples}


def _check_samples(source: ChromaSource, target: ActianTarget, coll: Any, opts: ValidateOptions,
                   transformer: RecordTransformer, sample_ids: list[str], distance: Distance) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    compared = 0
    vectors: list[np.ndarray] = []
    for i in range(0, len(sample_ids), 100):
        ids = sample_ids[i:i + 100]
        page = source.get_by_ids(coll, ids)
        expected: dict[str, Any] = {}
        for j, cid in enumerate(page.ids):
            try:
                point, _ = transformer.convert(cid, page.embeddings[j], page.documents[j] if page.documents is not None else None,
                                               page.metadatas[j] if page.metadatas is not None else None)
            except BadRecord:
                continue
            expected[canonical_id(point.id)] = (cid, point)
        if not expected:
            continue
        got = {canonical_id(p.id): p for p in target.get(opts.target_collection, [e[1].id for e in expected.values()],
                                                         with_payload=True, with_vectors=True)}
        for canon, (cid, point) in expected.items():
            compared += 1
            actual = got.get(canon)
            if actual is None:
                mismatches.append({"chroma_id": cid, "problem": "missing in target"})
                continue
            vec = np.asarray(point.vector, dtype=np.float32)
            vectors.append(vec)
            if not _vectors_match(vec, actual.vectors, distance):
                mismatches.append({"chroma_id": cid, "problem": "vector differs"})
            if _norm_payload(point.payload or {}) != _norm_payload(actual.payload or {}):
                mismatches.append({"chroma_id": cid, "problem": "payload differs",
                                   "expected_keys": sorted((point.payload or {}).keys()),
                                   "actual_keys": sorted((actual.payload or {}).keys())})
    ok = not mismatches and (compared > 0 or not sample_ids)
    return {"check": Check("sample_records", ok, {"compared": compared, "mismatches": len(mismatches),
                                                  "examples": mismatches[:10]}), "vectors": vectors}


def _load_golden(path: Path) -> list[tuple[str, list[float]]]:
    out: list[tuple[str, list[float]]] = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                out.append((str(rec.get("query_id", f"q{n}")), [float(x) for x in rec["embedding"]]))
            except (KeyError, ValueError, TypeError) as exc:
                raise UsageError(f"{path}:{n}: expected JSON {{'query_id', 'embedding': [...]}}: {exc}") from exc
    if not out:
        raise UsageError(f"{path} contains no queries")
    return out


def _check_search(source: ChromaSource, target: ActianTarget, coll: Any, opts: ValidateOptions,
                  sample_vectors: list[np.ndarray], dim: int | None, total: int, expected: int) -> Check:
    if expected == 0:
        return Check("search_parity", True, {"skipped": "empty collection"})
    kind = "golden" if opts.golden_queries else "self"
    if opts.golden_queries:
        queries = _load_golden(opts.golden_queries)
        bad = [q for q, v in queries if dim and len(v) != dim]
        if bad:
            return Check("search_parity", False, {"error": f"golden queries with wrong dimension: {bad[:5]}"})
    else:
        queries = [(f"self{i}", v.tolist()) for i, v in enumerate(sample_vectors[:opts.auto_queries])]
        if not queries:
            return Check("search_parity", False, {"error": "no sample vectors available to build queries"})
    k = min(opts.top_k, expected)
    # wait until the index returns a full result list (writes can be acknowledged before they are searchable)
    probe = queries[0][1]
    _poll(lambda: target.search(opts.target_collection, probe, k), lambda r: len(r) >= k, opts.poll_timeout)

    overlaps: list[float] = []
    hit1 = 0
    worst: list[dict[str, Any]] = []
    for qid, vec in queries:
        c_ids = [canonical_id(map_id(i, opts.transform.id_strategy, opts.transform.uuid_namespace))
                 for i in source.query(coll, vec, k)]
        a_ids = [canonical_id(r.id) for r in target.search(opts.target_collection, vec, k)]
        ov = len(set(c_ids) & set(a_ids)) / max(len(c_ids), 1)
        overlaps.append(ov)
        if c_ids and a_ids and c_ids[0] == a_ids[0]:
            hit1 += 1
        if ov < opts.min_query_overlap:
            worst.append({"query_id": qid, "overlap": round(ov, 3)})
    mean = float(np.mean(overlaps))
    lo = float(np.min(overlaps))
    ok = mean >= opts.min_mean_overlap and lo >= opts.min_query_overlap
    return Check("search_parity", ok, {
        "query_kind": kind, "queries": len(queries), "top_k": k, "mean_overlap": round(mean, 4),
        "min_overlap": round(lo, 4), "top1_agreement": round(hit1 / len(queries), 4),
        "thresholds": {"mean": opts.min_mean_overlap, "each": opts.min_query_overlap},
        "failing_queries": worst[:10],
        "note": "Both engines are approximate (HNSW) and break ties differently; thresholds are heuristics. "
                "Scores are not compared: Chroma returns distances, Actian returns scores."})
