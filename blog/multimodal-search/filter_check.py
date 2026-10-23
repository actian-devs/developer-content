from __future__ import annotations

import argparse
import time
import uuid

import numpy as np
from actian_vectorai import (
    Distance, Field, FilterBuilder, HnswConfigDiff, PointStruct, SearchParams,
    VectorAIClient, VectorParams,
)

POINT_COUNT = 200
INDEX_TIMEOUT_SECONDS = 120


def wait_for_index(client: VectorAIClient, collection: str) -> None:
    deadline = time.monotonic() + INDEX_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        stats = client.vde.get_stats(collection)
        if stats.total_vectors == POINT_COUNT and stats.indexed_vectors == POINT_COUNT:
            return
        time.sleep(0.5)
    stats = client.vde.get_stats(collection)
    raise TimeoutError(
        f"Timed out waiting for '{collection}' to index: "
        f"indexed {stats.indexed_vectors}/{POINT_COUNT}, "
        f"total {stats.total_vectors}/{POINT_COUNT}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether VectorAI applies text and boolean payload filters."
    )
    parser.add_argument("--url", default="localhost:6584")
    args = parser.parse_args()

    collection = f"filter_check_{uuid.uuid4().hex[:12]}"
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(POINT_COUNT, 32)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    expected_ids = set(range(0, POINT_COUNT, 2))
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"asset_id": "selected" if i in expected_ids else "other",
                     "is_selected": i in expected_ids},
        )
        for i in range(POINT_COUNT)
    ]

    with VectorAIClient(args.url) as client:
        client.collections.create(
            collection,
            vectors_config=VectorParams(
                size=vectors.shape[1],
                distance=Distance.Cosine,
                hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
            ),
        )
        try:
            for start in range(0, len(points), 100):
                client.points.upsert(collection, points=points[start : start + 100])
            if not client.vde.flush(collection):
                raise RuntimeError(f"VectorAI failed to flush collection '{collection}'")
            wait_for_index(client, collection)
            print(f"indexed {POINT_COUNT}/{POINT_COUNT}")

            tests = (
                ("text asset_id", Field("asset_id").eq("selected")),
                ("boolean is_selected", Field("is_selected").eq(True)),
            )
            all_ok = True
            for label, condition in tests:
                query_filter = FilterBuilder().must(condition).build()
                matched_count = client.points.count(collection, filter=query_filter)
                hits = client.points.search(
                    collection,
                    vectors[0].tolist(),
                    limit=POINT_COUNT,
                    filter=query_filter,
                    params=SearchParams(hnsw_ef=POINT_COUNT),
                )
                hit_ids = {int(hit.id) for hit in hits}
                off_filter = sorted(hit_ids - expected_ids)
                passed = matched_count == len(expected_ids) and not off_filter
                all_ok &= passed
                print(
                    f"{label}: matched {matched_count}/{len(expected_ids)}, "
                    f"returned {len(hit_ids)}, off-filter"
                    f": {off_filter if off_filter else 'none'} — "
                    f"{'PASS' if passed else 'FAIL'}"
                )

            if not all_ok:
                raise SystemExit("Payload filter check failed; do not publish benchmark results.")
        finally:
            client.collections.delete(collection, strict=False)


if __name__ == "__main__":
    main()
