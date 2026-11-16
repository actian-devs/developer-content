"""Run the must, should, must-not and min_should filters from the tutorial and re-check every result in Python."""
from __future__ import annotations

import argparse

from actian_vectorai import Field, FilterBuilder, VectorAIClient
from sentence_transformers import SentenceTransformer

from common import COLLECTION, URL, count_matching, load_catalog


def show(title, hits, rule):
    print(f"\n{title}")
    for h in hits:
        p = h.payload
        print(f"  {'ok ' if rule(p) else 'BAD'} {h.score:.3f}  {p['title']:<38} {p['category']:<12} "
              f"${p['price']:<7} rating {p['rating']}  {p['condition']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--query", default="noise cancelling headphones for travel")
    args = ap.parse_args()

    _, _, _, info = load_catalog()
    q = SentenceTransformer(info["model"]).encode(args.query, normalize_embeddings=True).tolist()

    combined = (
        FilterBuilder()
        .must(Field("category").eq("electronics"))        # has to be electronics
        .should(Field("price").lt(50.0))                  # can be cheap
        .should(Field("rating").gt(4.0))                  # or highly rated
        .must_not(Field("condition").eq("refurbished"))   # cannot be refurbished
        .build()
    )
    two_of_three = (
        FilterBuilder()
        .must(Field("category").eq("electronics"))
        .min_should([Field("price").lt(50.0), Field("rating").gt(4.0), Field("in_stock").eq(True)], 2)
        .build()
    )

    with VectorAIClient(args.url) as client:
        for title, flt, rule in [
            ("must electronics, should cheap or highly rated, must not refurbished", combined,
             lambda p: p["category"] == "electronics" and (p["price"] < 50 or p["rating"] > 4.0)
             and p["condition"] != "refurbished"),
            ("must electronics, at least two of cheap, highly rated, in stock", two_of_three,
             lambda p: p["category"] == "electronics"
             and sum([p["price"] < 50, p["rating"] > 4.0, p["in_stock"]]) >= 2),
        ]:
            admitted = count_matching(client, args.collection, flt)
            hits = client.points.search(args.collection, q, limit=5, filter=flt, with_payload=True)
            show(f"{title}  ({admitted} products admitted, {len(hits)} returned)", hits, rule)
            if len(hits) < min(5, admitted):
                print(f"  recall loss: {len(hits)} returned although {admitted} products match")


if __name__ == "__main__":
    main()