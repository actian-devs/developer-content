"""Read-only scan of a Chroma collection: what will migrate cleanly, what will be changed, what will block."""

from __future__ import annotations

import logging
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from actian_vectorai import Distance
from actian_vectorai.utils.validation import validate_collection_name

from .errors import UsageError
from .progress import Progress
from .source import ChromaSource
from .state import write_json
from .transform import (MAX_DIMENSION, RecordTransformer, TransformOptions, resolve_distance)

log = logging.getLogger(__name__)

BLOCKING, WARNING, INFO = "BLOCKING", "WARNING", "INFO"


@dataclass
class PreflightOptions:
    transform: TransformOptions
    target_collection: str
    distance_override: str | None = None
    on_bad_record: str = "fail"
    allow_zero_vectors: bool = False
    workdir: Path = Path("migration_work")
    progress_interval: float = 10.0
    max_examples: int = 10


def _value_type(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (list, tuple)):
        return "array"
    return type(v).__name__


def _id_kind(i: str) -> str:
    if i.isascii() and i.isdigit() and (i == "0" or not i.startswith("0")):
        return "int"
    try:
        return "uuid" if str(uuid.UUID(i)) == i else "other"
    except ValueError:
        return "other"


def run_preflight(source: ChromaSource, collection: str, opts: PreflightOptions) -> dict[str, Any]:
    coll = source.get_collection(collection)
    total = source.count(coll)
    space, space_src = source.space(coll)
    distance = resolve_distance(space, opts.distance_override)

    findings: list[dict[str, str]] = []

    def add(level: str, code: str, msg: str) -> None:
        findings.append({"level": level, "code": code, "message": msg})

    try:
        validate_collection_name(opts.target_collection)
    except Exception as exc:  # noqa: BLE001
        add(BLOCKING, "target_name_invalid", f"target collection name {opts.target_collection!r}: {exc}")

    if space_src == "default":
        add(WARNING, "distance_assumed",
            "Could not read the distance space from the collection; assuming Chroma's default (l2 -> Euclid). "
            "Pass --distance cosine|euclid|dot if that is wrong.")

    report: dict[str, Any] = {
        "collection": collection, "target_collection": opts.target_collection, "count": total,
        "chroma_space": space, "chroma_space_source": space_src, "actian_distance": distance.name,
        "transform": opts.transform.fingerprint(),
    }

    if total == 0:
        add(WARNING, "empty_collection", "collection is empty: there is nothing to migrate")
        report["findings"] = findings
        return _finish(report, findings, opts)

    topts = opts.transform
    topts.reject_zero_vectors = distance == Distance.Cosine and not opts.allow_zero_vectors
    transformer: RecordTransformer | None = None
    dims: Counter[int] = Counter()
    id_kinds: Counter[str] = Counter()
    max_id_len = 0
    key_types: dict[str, Counter[str]] = defaultdict(Counter)
    docs_none = docs_empty = 0
    doc_bytes = doc_max = 0
    meta_none_records = 0
    reject_reasons: Counter[str] = Counter()
    reject_examples: list[dict[str, Any]] = []
    scanned = 0
    prog = Progress(total, label="scanned", interval=opts.progress_interval)

    for page in source.iter_pages(coll, include=("embeddings", "documents", "metadatas")):
        if transformer is None:
            first = page.embeddings[0] if page.embeddings is not None and len(page.embeddings) else None
            dim0 = int(len(first)) if first is not None else None
            transformer = RecordTransformer(topts, dimension=dim0)
            report["dimension"] = dim0
        for i, cid in enumerate(page.ids):
            emb = page.embeddings[i] if page.embeddings is not None else None
            if emb is not None:
                dims[int(len(emb))] += 1
            id_kinds[_id_kind(cid)] += 1
            max_id_len = max(max_id_len, len(cid))
            doc = page.documents[i] if page.documents is not None else None
            if doc is None:
                docs_none += 1
            else:
                if doc == "":
                    docs_empty += 1
                b = len(doc.encode("utf-8"))
                doc_bytes += b
                doc_max = max(doc_max, b)
            md = page.metadatas[i] if page.metadatas is not None else None
            if md is None:
                meta_none_records += 1
            else:
                for k, v in md.items():
                    key_types[str(k)][_value_type(v)] += 1
        converted = transformer.convert_page(page)
        for r in converted.rejects:
            reject_reasons[r.reason] += 1
            if len(reject_examples) < opts.max_examples:
                reject_examples.append({"chroma_id": r.chroma_id, "reason": r.reason})
        scanned += len(page.ids)
        prog.advance(len(page.ids))

    rejected = sum(reject_reasons.values())
    report.update({
        "scanned": scanned, "dimensions": {str(k): v for k, v in dims.items()},
        "id_kinds": dict(id_kinds), "max_id_length": max_id_len,
        "documents": {"none": docs_none, "empty": docs_empty, "total_bytes": doc_bytes, "max_bytes": doc_max},
        "records_without_metadata": meta_none_records,
        "metadata_keys": {k: dict(v) for k, v in sorted(key_types.items())},
        "rejects": {"total": rejected, "reasons": dict(reject_reasons), "examples": reject_examples},
        "transform_counters": dict(transformer.counters) if transformer else {},
        "metadata_key_renames": dict(transformer.field_map) if transformer else {},
    })

    if scanned != total:
        add(BLOCKING if abs(scanned - total) else INFO, "count_changed",
            f"scanned {scanned} records but count() reported {total}: the collection is being written to. "
            "Freeze writes before migrating.")
    if len(dims) > 1:
        add(BLOCKING, "mixed_dimensions", f"records have different embedding dimensions: {dict(dims)}")
    dim = report.get("dimension")
    if dim and dim > MAX_DIMENSION:
        add(BLOCKING, "dimension_too_large", f"dimension {dim} exceeds Actian's maximum of {MAX_DIMENSION}")
    if rejected:
        level = BLOCKING if opts.on_bad_record == "fail" else WARNING
        suffix = ("" if opts.on_bad_record == "fail"
                  else " They will be skipped and listed in rejected.jsonl (--on-bad-record skip).")
        add(level, "records_rejected", f"{rejected} record(s) cannot be converted: {dict(reject_reasons)}.{suffix}")
    if transformer and transformer.field_map:
        add(WARNING, "metadata_keys_renamed",
            f"metadata keys collide with reserved payload fields and will be renamed: {transformer.field_map}. "
            "Update application filters/reads accordingly.")
    if transformer and transformer.counters.get("int_out_of_int64_stringified"):
        add(WARNING, "int_stringified", f"{transformer.counters['int_out_of_int64_stringified']} metadata integers "
            "exceed int64 and will be stored as strings")
    mixed = {k: dict(v) for k, v in key_types.items() if len({t for t in v if t != "null"}) > 1}
    if mixed:
        add(WARNING, "mixed_type_keys", f"metadata keys with mixed value types (filters may not match across types): {mixed}")
    if docs_none:
        add(INFO, "documents_missing", f"{docs_none} record(s) have no document; the document field is omitted for them")
    if report["transform"]["id_strategy"] == "uuid5" and id_kinds:
        add(INFO, "ids_remapped", "all IDs become deterministic UUIDv5 values; the original Chroma ID is kept in the "
            f"payload field {topts.id_field!r}")
    if dim:
        raw_gib = total * dim * 4 / 2**30
        report["raw_vector_gib"] = round(raw_gib, 3)
        add(INFO, "size_estimate", f"raw float32 vectors: {raw_gib:.2f} GiB for {total} x {dim}; "
            "the HNSW index and payloads add to this. Check your Actian licence/capacity limit against the record count.")

    report["findings"] = findings
    return _finish(report, findings, opts)


def _finish(report: dict[str, Any], findings: list[dict[str, str]], opts: PreflightOptions) -> dict[str, Any]:
    blocking = sum(1 for f in findings if f["level"] == BLOCKING)
    report["ok"] = blocking == 0
    write_json(opts.workdir / "preflight.json", report)
    for f in findings:
        log.log(logging.ERROR if f["level"] == BLOCKING else logging.WARNING if f["level"] == WARNING else logging.INFO,
                "[%s] %s: %s", f["level"], f["code"], f["message"])
    return report
