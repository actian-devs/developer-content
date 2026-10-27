"""Blocking with VectorAI DB: each left record's K nearest right records become its candidates.

A filter inside the search restricts it to the other source, so a record is never a candidate
for another record from its own table. On customer or supplier data, the same `must` clause is
where rules that rule a pair out completely go (a different country, a person against a company).
"""
import numpy as np
from actian_vectorai import (Distance, Field, FilterBuilder, PointStruct, SearchParams, VectorAIClient,
                             VectorParams)

UPSERT_BATCH = 500
SEARCH_BATCH = 100   # the SDK's search_batch limit


def load_records(db: VectorAIClient, collection: str, ds, vectors: np.ndarray) -> None:
    """Recreate the collection and upsert every record with its source and ID in the payload.

    The Community edition caps the whole server at 5,000 vectors across all collections, so the
    harness first drops its own collections from earlier runs (names starting with "er_"). Each run
    rebuilds its collection from the cached embeddings, so nothing is lost.
    """
    for name in db.collections.list():
        if name.startswith("er_") and name != collection:
            print(f"  dropping {name} from an earlier run (Community edition counts all collections together)")
            db.collections.delete(name)
    if db.collections.exists(collection):
        db.collections.delete(collection)
    try:
        db.collections.create(collection, vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.Cosine))
    except Exception as e:
        if "capacity" in str(e).lower() or "limit exceeded" in str(e).lower():
            # observed on 1.0.2: the vector count does not drop when collections are deleted until a restart
            raise SystemExit("VectorAI DB still counts the deleted collections against the 5,000-vector limit. "
                             "Restart it (docker compose restart vectorai) and run again.") from e
        raise

    rows = [(ds.spec.left_name, r) for _, r in ds.left.iterrows()] + \
           [(ds.spec.right_name, r) for _, r in ds.right.iterrows()]
    points = [PointStruct(id=i, vector=v.tolist(),
                          payload={"source": src, "record_id": r["id"], "text": r["text"]})
              for i, ((src, r), v) in enumerate(zip(rows, vectors))]
    for start in range(0, len(points), UPSERT_BATCH):
        db.points.upsert(collection, points[start:start + UPSERT_BATCH])


def _search(db, collection, vectors, limit, flt, params) -> list[list]:
    out = []
    for start in range(0, len(vectors), SEARCH_BATCH):   # search_batch takes up to 100 queries per call
        searches = [{"vector": v.tolist(), "limit": limit, "filter": flt, "with_payload": True, "params": params}
                    for v in vectors[start:start + SEARCH_BATCH]]
        out.extend(db.points.search_batch(collection, searches))
    return out


def neighbours(db: VectorAIClient, collection: str, ds, left_vectors: np.ndarray, k: int,
               hnsw_ef: int, oversample: int = 10) -> dict[str, list]:
    """Return {left_id: [(right_id, similarity), ...]}, nearest first, for each left record's k nearest right records.

    On the test runs, a filtered search for `limit` results returned only the matches for the
    filter among roughly the `limit` nearest records overall. A record's nearest vectors are
    mostly from its own table, so asking for k gave back fewer than k. We over-fetch (k x
    oversample), keep the top k, and retry any record still short with a larger limit.
    """
    other_source = FilterBuilder().must(Field("source").eq(ds.spec.right_name)).build()
    params = SearchParams(hnsw_ef=hnsw_ef) if hnsw_ef else None
    total = len(ds.left) + len(ds.right)
    left_ids = list(ds.left["id"])
    ranked, todo, limit = {}, list(range(len(left_ids))), min(k * oversample, total)
    while todo:
        results = _search(db, collection, left_vectors[todo], limit, other_source, params)
        short = []
        for i, hits in zip(todo, results):
            ranked[left_ids[i]] = sorted(((h.payload["record_id"], float(h.score)) for h in hits),
                                         key=lambda x: -x[1])[:k]
            if len(hits) < k and limit < total:
                short.append(i)
        todo, limit = short, min(limit * 4, total)
    return ranked


def top_k(ranked: dict[str, list], k: int) -> dict[tuple[str, str], float]:
    """Candidate pairs {(left_id, right_id): similarity} from the k nearest neighbours of each left record."""
    return {(left_id, right_id): score for left_id, hits in ranked.items() for right_id, score in hits[:k]}


def returned_per_record(ranked: dict[str, list], k: int) -> dict:
    """How many of the k requested neighbours each left record actually got back."""
    counts = [min(len(hits), k) for hits in ranked.values()]
    return {"requested": k, "mean": float(np.mean(counts)), "min": int(min(counts)),
            "records_short": sum(c < k for c in counts), "records": len(counts)}


def blocking_quality(candidates: dict, ds) -> dict:
    """Reduction ratio: share of all pairs removed. Pair completeness: share of true matches kept."""
    all_pairs = len(ds.left) * len(ds.right)
    kept = len(ds.gold & candidates.keys())
    return {"all_pairs": all_pairs, "candidate_pairs": len(candidates),
            "reduction_ratio": 1 - len(candidates) / all_pairs,
            "pair_completeness": kept / len(ds.gold), "true_matches_kept": kept}
