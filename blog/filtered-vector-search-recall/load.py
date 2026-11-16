"""Create the products collection in VectorAI DB and upsert the catalog."""
from __future__ import annotations

import argparse

from actian_vectorai import Distance, PointStruct, VectorAIClient, VectorParams

from common import COLLECTION, URL, load_catalog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()

    vecs, _, products, info = load_catalog()
    with VectorAIClient(args.url) as client:
        if client.collections.exists(args.collection):
            client.collections.delete(args.collection)
        client.collections.create(args.collection,
                                  vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.Cosine))
        points = [PointStruct(id=i, vector=vecs[i].tolist(), payload=products[i]) for i in range(len(products))]
        for start in range(0, len(points), 256):          # small batches keep each gRPC message small
            client.points.upsert(args.collection, points=points[start : start + 256])
        client.vde.flush(args.collection)
        client.vde.wait_for_ready(args.collection)         # do not search a collection that is still indexing
        print(f"loaded {len(products)} products ({vecs.shape[1]}-dimensional, {info['model']}) "
              f"into '{args.collection}'")


if __name__ == "__main__":
    main()