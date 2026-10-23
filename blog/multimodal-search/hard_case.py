from __future__ import annotations

import argparse
import json
from pathlib import Path

from actian_vectorai import Field, FilterBuilder, VectorAIClient

from embed import embed_media, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--collection", default="field_search")
    ap.add_argument("--url", default="localhost:6584")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    photo_queries = [q for q in json.loads(Path(args.queries).read_text())
                     if q["query_modality"] == "photo" and q["target_modality"] == "manual_page"]
    model = load_model()
    flt = FilterBuilder().must(Field("modality").eq("manual_page")).build()

    hit_scores, miss_scores, misses = [], [], 0
    with VectorAIClient(args.url) as client:
        for rec in photo_queries:
            vec = embed_media(model, "image", rec["query"]).tolist()
            hits = client.points.search(args.collection, vec, limit=args.k, filter=flt, with_payload=True)
            if any(h.payload["asset_id"] in rec["relevant"] for h in hits):
                hit_scores.append(hits[0].score)
                continue
            misses += 1
            miss_scores.append(hits[0].score if hits else float("nan"))
            print(f"\nMISS {rec['query']}  expected {rec['relevant']}")
            for h in hits:
                print(f"  {h.score:.4f}  {h.payload['asset_id']:<14} {h.payload.get('path', '')}")

    n = len(photo_queries)
    print(f"\n{misses} misses out of {n} photo -> manual_page queries")
    if hit_scores and miss_scores:
        print(f"top-1 score on hits:   min {min(hit_scores):.4f}  max {max(hit_scores):.4f}")
        print(f"top-1 score on misses: min {min(miss_scores):.4f}  max {max(miss_scores):.4f}")


if __name__ == "__main__":
    main()