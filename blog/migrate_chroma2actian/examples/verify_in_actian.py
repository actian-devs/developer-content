#!/usr/bin/env python3
"""Independent spot-check of a migrated collection, straight from Actian (no migration code involved).

    python examples/verify_in_actian.py --collection support_tickets_demo --expect 500
"""
from __future__ import annotations

import argparse
import sys

from actian_vectorai import Field, FilterBuilder, VectorAIClient


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=None, help="host:port (default: ACTIAN_VECTORAI_URL or localhost:6574)")
    p.add_argument("--collection", required=True)
    p.add_argument("--expect", type=int, required=True, help="number of records in the Chroma source")
    a = p.parse_args()

    with (VectorAIClient(a.url) if a.url else VectorAIClient()) as client:
        n = client.points.count(a.collection)
        print(f"count: {n} (expected {a.expect}) -> {'OK' if n == a.expect else 'MISMATCH'}")

        pts, _ = client.points.scroll(a.collection, limit=1, with_payload=True, with_vectors=True)
        if not pts:
            print("collection is empty")
            return 1
        p0 = pts[0]
        print(f"sample point id={p0.id}\n  payload={p0.payload}\n  vector dims={len(p0.vectors)}")

        hits = client.points.search(a.collection, vector=p0.vectors, limit=3, with_payload=True)
        print("nearest to the sample (first hit should be itself):")
        for h in hits:
            print(f"  id={h.id} score={h.score:.4f} chroma_id={(h.payload or {}).get('chroma_id')}")

        f = FilterBuilder().must(Field("resolved").eq(True)).build()
        print(f"payload filter resolved==true -> {client.points.count(a.collection, filter=f)} points")
    return 0 if n == a.expect else 1


if __name__ == "__main__":
    sys.exit(main())
