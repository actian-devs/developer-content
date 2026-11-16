"""Recall at k for a client-side post-filter and for VectorAI DB filtering, across filter selectivities.

Ground truth is exact filtered search in NumPy over the same vectors. Each filter is a price ceiling
chosen so it admits the requested share of the catalog. Methods per share:

  post_filter           unfiltered points.search for the top k, then drop results that fail the filter
  vectorai_filter       points.search with FilterBuilder().must(Field("price").lte(ceiling)), server defaults
  vectorai_ef<N>        the same filtered search with SearchParams(hnsw_ef=N)
  vectorai_overfetch<F> the same filtered search with limit = k * F, truncated to k

Every row reports recall at k, the average number of results returned (of k), and how many returned
results broke the filter, so a filter that was not applied shows up as a count and not as low recall.
--probe-limits adds a diagnostic that prints how many filtered results come back as the limit grows.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from actian_vectorai import Field, FilterBuilder, SearchParams, VectorAIClient

from common import COLLECTION, URL, count_matching, load_catalog, versions


def recall(found: set, truth: set) -> float:
    return len(found & truth) / max(len(truth), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--shares", type=float, nargs="+", default=[0.5, 0.2, 0.1, 0.05, 0.01])
    ap.add_argument("--hnsw-ef", type=int, nargs="*", default=[256])
    ap.add_argument("--overfetch", type=int, nargs="*", default=[20])
    ap.add_argument("--probe-limits", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="recall.csv")
    ap.add_argument("--hardware", required=True, help='for example "AMD Ryzen 5 2500U, 8 GB RAM, no CUDA GPU"')
    args = ap.parse_args()

    vecs, qvecs, products, info = load_catalog()
    prices = np.array([p["price"] for p in products])
    k = args.k
    print(f"Hardware: {args.hardware} | {versions()} | embeddings: {info['model']} | "
          f"{len(products)} products, {len(qvecs)} queries | k={k}\n")

    rows, probe_rows = [], []
    with VectorAIClient(args.url) as client:
        for share in args.shares:
            ceiling = float(np.quantile(prices, share))
            mask = prices <= ceiling
            flt = FilterBuilder().must(Field("price").lte(ceiling)).build()
            admitted = count_matching(client, args.collection, flt)
            flag = "" if admitted == int(mask.sum()) else "   MISMATCH, check the collection"
            print(f"filter price <= {ceiling:.2f}: admits {int(mask.sum())} products ({mask.mean():.1%}); "
                  f"the server counts {admitted}{flag}")
            allowed = np.flatnonzero(mask)

            methods = ["post_filter", "vectorai_filter"]
            methods += [f"vectorai_ef{e}" for e in args.hnsw_ef]
            methods += [f"vectorai_overfetch{f}" for f in args.overfetch]
            stats = {m: {"r": [], "n": [], "bad": 0} for m in methods}

            for q in qvecs:
                truth = {int(i) for i in allowed[np.argsort(-(vecs[allowed] @ q))][:k]}
                qv = q.tolist()
                found = {}
                hits = client.points.search(args.collection, qv, limit=k, with_payload=True)
                found["post_filter"] = [h for h in hits if h.payload["price"] <= ceiling]
                found["vectorai_filter"] = client.points.search(args.collection, qv, limit=k, filter=flt,
                                                                with_payload=True)
                for e in args.hnsw_ef:
                    found[f"vectorai_ef{e}"] = client.points.search(
                        args.collection, qv, limit=k, filter=flt, params=SearchParams(hnsw_ef=e), with_payload=True)
                for f in args.overfetch:
                    found[f"vectorai_overfetch{f}"] = client.points.search(
                        args.collection, qv, limit=k * f, filter=flt, with_payload=True)[:k]
                for m in methods:
                    ids = {int(h.payload["idx"]) for h in found[m]}
                    stats[m]["r"].append(recall(ids, truth))
                    stats[m]["n"].append(len(found[m]))
                    stats[m]["bad"] += sum(1 for h in found[m] if h.payload["price"] > ceiling)

            for m in methods:
                s = stats[m]
                row = {"share": share, "ceiling": round(ceiling, 2), "admitted": int(mask.sum()), "method": m,
                       "recall_at_k": round(float(np.mean(s["r"])), 3), "returned": round(float(np.mean(s["n"])), 1),
                       "off_filter": s["bad"]}
                rows.append(row)
                print(f"  {m:<24} recall@{k} {row['recall_at_k']:.3f}   returned {row['returned']:>5}   "
                      f"off-filter {row['off_filter']}")

            for lim in args.probe_limits:
                counts = [len(client.points.search(args.collection, q.tolist(), limit=lim, filter=flt))
                          for q in qvecs[:20]]
                probe_rows.append({"share": share, "limit": lim, "returned": round(float(np.mean(counts)), 1)})
                print(f"  limit probe: limit={lim} returned {np.mean(counts):.1f} filtered results "
                      f"(the filter admits {int(mask.sum())})")
            print()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if probe_rows:
        with open("probe.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(probe_rows[0]))
            w.writeheader()
            w.writerows(probe_rows)
    print(f"wrote {args.out}" + (" and probe.csv" if probe_rows else ""))


if __name__ == "__main__":
    main()