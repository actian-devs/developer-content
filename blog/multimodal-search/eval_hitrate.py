from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from actian_vectorai import Field, FilterBuilder, VectorAIClient

from embed import embed_media, embed_query, load_model


def embed_for(model, rec):
    qm = rec["query_modality"]
    if qm == "text":
        return embed_query(model, rec["query"])
    return embed_media(
        model,
        {"photo": "image", "voice": "audio", "video": "video"}[qm],
        rec["query"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--collection", default="field_search")
    ap.add_argument("--url", default="localhost:6584")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="hitrate.csv")
    args = ap.parse_args()

    queries = json.loads(Path(args.queries).read_text())
    model = load_model()
    tally = defaultdict(lambda: [0, 0])  # pair -> [hits, total]

    with VectorAIClient(args.url) as client:
        for rec in queries:
            flt = FilterBuilder().must(Field("modality").eq(rec["target_modality"])).build()
            hits = client.points.search(
                args.collection, embed_for(model, rec).tolist(),
                limit=args.k, filter=flt, with_payload=True,
            )
            ok = any(h.payload["asset_id"] in rec["relevant"] for h in hits)
            key = (rec["query_modality"], rec["target_modality"])
            tally[key][0] += int(ok)
            tally[key][1] += 1

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_modality", "target_modality", f"hit_rate_at_{args.k}", "queries"])
        for (qm, tm), (hit, total) in sorted(tally.items()):
            w.writerow([qm, tm, f"{hit / total:.3f}", total])
            print(f"{qm:>6} -> {tm:<12} hit@{args.k} = {hit / total:.3f}  ({hit}/{total})")


if __name__ == "__main__":
    main()