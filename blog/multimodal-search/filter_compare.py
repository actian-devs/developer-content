from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np
from actian_vectorai import (
    Distance, Field, FilterBuilder, HnswConfigDiff, PointStruct, SearchParams,
    VectorAIClient, VectorParams,
)

HNSW_M = 32
EF_CONSTRUCT = 200
EF_SEARCH = 16
INDEX_TIMEOUT_SECONDS = 120


def p50_ms(times: list[float]) -> float:
    return float(np.percentile(times, 50) * 1000)


def wait_for_index(client: VectorAIClient, name: str, expected: int) -> None:
    deadline = time.monotonic() + INDEX_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        stats = client.vde.get_stats(name)
        if stats.total_vectors == expected and stats.indexed_vectors == expected:
            return
        time.sleep(0.5)
    stats = client.vde.get_stats(name)
    raise TimeoutError(
        f"Timed out waiting for '{name}' to index: "
        f"indexed {stats.indexed_vectors}/{expected}, total {stats.total_vectors}/{expected}"
    )


def score_searches(
    queries: np.ndarray,
    truth: list[set[int]],
    allowed: set[int],
    search: Callable[[np.ndarray], list[int]],
) -> tuple[float, float, float, int]:
    recalls, times, returned_counts = [], [], []
    off_filter = 0
    for query, expected in zip(queries, truth):
        started = time.perf_counter()
        ids = [int(point_id) for point_id in search(query) if int(point_id) >= 0]
        times.append(time.perf_counter() - started)
        found = set(ids)
        off_filter += sum(point_id not in allowed for point_id in ids)
        returned_counts.append(len(ids))
        recalls.append(len(found & expected) / max(len(expected), 1))
    return (
        float(np.mean(recalls)),
        p50_ms(times),
        float(np.mean(returned_counts)),
        off_filter,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--filter-share", type=float, default=0.01,
                    help="fraction of each sampled corpus admitted by the payload filter")
    ap.add_argument("--url", default="localhost:6584")
    ap.add_argument("--data-dir", default="./vectorai_data",
                    help="accepted for CLI compatibility; index sizes come from VectorAI stats")
    ap.add_argument("--hardware", required=True, help='e.g. "Ryzen 7 5800X, 32GB RAM, CPU-only"')
    args = ap.parse_args()
    if not 0 < args.filter_share <= 1:
        ap.error("--filter-share must be greater than 0 and at most 1")
    if any(size < 1 for size in args.sizes) or args.k < 1 or args.queries < 1:
        ap.error("--sizes, --k, and --queries must all be positive")

    print("filter_compare v3")
    try:
        vecs = np.load("filter_embeddings.npy").astype("float32")
        meta = json.loads(Path("filter_meta.json").read_text())
    except FileNotFoundError as exc:
        raise SystemExit(
            "Filter benchmark cache not found. Prepare it first with "
            f"`uv run filter_corpus.py --source filter_text --target {max(args.sizes)} "
            "--chunk-words 10`."
        ) from exc
    if vecs.ndim != 2 or len(meta) != len(vecs):
        raise SystemExit(
            "Filter benchmark cache is inconsistent: filter_embeddings.npy and "
            "filter_meta.json must contain the same number of vectors and metadata rows."
        )
    if max(args.sizes) > len(vecs):
        raise SystemExit(
            f"Requested corpus size {max(args.sizes)} exceeds the {len(vecs)} vectors in "
            "filter_embeddings.npy. Prepare a larger cache with "
            f"`uv run filter_corpus.py --source filter_text --target {max(args.sizes)} "
            "--chunk-words 10`."
        )
    if max(args.sizes) < args.queries:
        ap.error("--queries cannot exceed the smallest requested corpus size")

    rng = np.random.default_rng(0)
    print(
        f"Hardware: {args.hardware}\n\n"
        f"Vectors: {len(vecs)} from filter_embeddings.npy | "
        f"filter admits {args.filter_share:.0%} | k={args.k} | "
        f"HNSW M={HNSW_M}, efSearch={EF_SEARCH}"
    )

    with VectorAIClient(args.url) as client:
        for size in args.sizes:
            ids = rng.choice(len(vecs), size, replace=False)
            sub = vecs[ids]

            allowed_count = max(1, int(round(size * args.filter_share)))
            allowed_ids = np.sort(rng.choice(size, allowed_count, replace=False)).astype("int64")
            allowed = set(allowed_ids.tolist())
            queries = sub[rng.choice(size, args.queries, replace=False)]
            truth = [
                set(allowed_ids[np.argsort(-(sub[allowed_ids] @ query))][: args.k].tolist())
                for query in queries
            ]

            index = faiss.IndexHNSWFlat(sub.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = EF_CONSTRUCT
            index.hnsw.efSearch = EF_SEARCH
            index.add(sub)
            index_bytes = faiss.serialize_index(index).nbytes
            rows = []

            def post_filter_search(query: np.ndarray) -> list[int]:
                _, found_ids = index.search(query[None], args.k)
                return [int(i) for i in found_ids[0] if int(i) in allowed]

            rows.append((
                "faiss_hnsw_post_filter",
                score_searches(queries, truth, allowed, post_filter_search),
                index_bytes,
                None,
            ))

            selector = faiss.IDSelectorBatch(allowed_ids)
            selector_params = faiss.SearchParametersHNSW()
            selector_params.efSearch = EF_SEARCH
            selector_params.sel = selector

            def selector_search(query: np.ndarray) -> list[int]:
                _, found_ids = index.search(query[None], args.k, params=selector_params)
                return [int(i) for i in found_ids[0]]

            rows.append((
                "faiss_hnsw_id_selector",
                score_searches(queries, truth, allowed, selector_search),
                index_bytes,
                None,
            ))

            name = f"filter_compare_{size}"
            client.collections.delete(name, strict=False)
            client.collections.create(
                name,
                vectors_config=VectorParams(
                    size=sub.shape[1],
                    distance=Distance.Cosine,
                    hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=EF_CONSTRUCT),
                ),
            )
            try:
                points = [
                    PointStruct(
                        id=i,
                        vector=sub[i].tolist(),
                        payload={"asset_id": "selected" if i in allowed else "other"},
                    )
                    for i in range(size)
                ]
                for start in range(0, len(points), 256):
                    client.points.upsert(name, points=points[start : start + 256])
                if not client.vde.flush(name):
                    raise RuntimeError(f"VectorAI failed to flush collection '{name}'")
                wait_for_index(client, name, size)

                flt = FilterBuilder().must(Field("asset_id").eq("selected")).build()
                search_params = SearchParams(hnsw_ef=EF_SEARCH)

                def vectorai_search(query: np.ndarray) -> list[int]:
                    hits = client.points.search(
                        name,
                        query.tolist(),
                        limit=args.k,
                        filter=flt,
                        params=search_params,
                    )
                    return [int(hit.id) for hit in hits]

                vectorai_result = score_searches(
                    queries, truth, allowed, vectorai_search
                )
                stats = client.vde.get_stats(name)
                rows.append((
                    "vectorai",
                    vectorai_result,
                    stats.index_memory_bytes,
                    stats.indexed_vectors,
                ))
            finally:
                client.collections.delete(name, strict=False)

            print(
                f"corpus {size}  filter admits {allowed_count} vectors "
                f"({allowed_count / size:.1%})"
            )
            for label, (recall, latency, returned, off_filter), nbytes, indexed in rows:
                indexed_text = f"   indexed {indexed}/{size}" if indexed is not None else ""
                print(
                    f"  {label:<25} recall@{args.k} {recall:.3f}   "
                    f"returned {returned:.1f}/{args.k}   off-filter {off_filter}   "
                    f"p50 {latency:.3f} ms   index {nbytes / 1e6:.1f} MB"
                    f"{indexed_text}"
                )


if __name__ == "__main__":
    main()
