# query.py
from __future__ import annotations

import argparse
import time

from actian_vectorai import VectorAIClient, FilterBuilder, Field

from clip_encoder import ClipEncoder

COLLECTION = "defect_search"   # must match the name ingest.py created


def build_filter(part_category: str | None = None, defect_only: bool = False):
    """Metadata filter, applied inside the search rather than after it."""
    builder = FilterBuilder()
    if part_category:
        builder = builder.must(Field("part_category").eq(part_category))
    if defect_only:
        builder = builder.must(Field("is_defect").eq(True))
    return None if builder.is_empty() else builder.build()


def search_by_text(client, encoder, text, part_category=None, defect_only=False, limit=5):
    vector = encoder.encode_text([text])[0].tolist()
    return client.points.search(
        COLLECTION, vector, limit=limit,
        filter=build_filter(part_category, defect_only), with_payload=True,
    )


def search_by_image(client, encoder, image_path, part_category=None, defect_only=False, limit=5):
    vector = encoder.encode_images([image_path])[0].tolist()
    return client.points.search(
        COLLECTION, vector, limit=limit,
        filter=build_filter(part_category, defect_only), with_payload=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--image")
    ap.add_argument("--part-category")
    ap.add_argument("--defect-only", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    if not args.text and not args.image:
        raise SystemExit("Pass --text or --image")

    encoder = ClipEncoder()
    with VectorAIClient("localhost:6574") as client:
        t0 = time.perf_counter()
        if args.text:
            results = search_by_text(client, encoder, args.text,
                                     args.part_category, args.defect_only, args.limit)
        else:
            results = search_by_image(client, encoder, args.image,
                                      args.part_category, args.defect_only, args.limit)
        total = (time.perf_counter() - t0) * 1000

    print(f"  score  {'category':<12} {'defect':<16} path")
    for r in results:
        p = r.payload
        print(f"{r.score:7.4f}  {p['part_category']:<12} {p['defect_type']:<16} {p['image_path']}")
    print(f"\ntotal {total:.1f} ms")


if __name__ == "__main__":
    main()