from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from actian_vectorai import Field, FilterBuilder, VectorAIClient

from embed import embed_media, embed_query, load_model

MEDIA_KIND = {"photo": "image", "voice": "audio", "video": "video"}


def embed_for(model, rec):
    modality = rec["query_modality"]
    if modality == "text":
        return embed_query(model, rec["query"])
    try:
        kind = MEDIA_KIND[modality]
    except KeyError as exc:
        raise ValueError(f"Unsupported query modality: {modality}") from exc
    return embed_media(model, kind, rec["query"])


def describe_hit(hit) -> str:
    payload = hit.payload or {}
    asset_id = payload.get("asset_id", "<missing asset_id>")
    modality = payload.get("modality", "<missing modality>")
    location = payload.get("path") or payload.get("source") or ""
    text = " ".join(str(payload.get("text", "")).split())
    details = f"{asset_id} [{modality}]"
    if location:
        details += f"  {location}"
    if text:
        details += f"  {text[:180]}{'...' if len(text) > 180 else ''}"
    return details


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Show queries whose relevant asset is missing from the top-k results."
    )
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--collection", default="field_search")
    ap.add_argument("--url", default="localhost:6584")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--query-modality",
        choices=("text", "photo", "voice", "video"),
        help="evaluate only this query modality (default: evaluate all query modalities)",
    )
    args = ap.parse_args()
    if args.k < 1:
        ap.error("--k must be positive")

    queries = json.loads(Path(args.queries).read_text())
    selected = [
        rec for rec in queries
        if args.query_modality is None or rec["query_modality"] == args.query_modality
    ]
    if not selected:
        raise SystemExit("No queries matched the requested query modality.")

    tally = defaultdict(lambda: [0, 0])
    misses = 0
    with VectorAIClient(args.url) as client:
        if not client.collections.exists(args.collection):
            raise SystemExit(
                f"Collection '{args.collection}' does not exist. Restore it with "
                "`uv run ingest.py --from-cache`. If the cache does not match "
                "manifest.json, rebuild it with "
                "`uv run ingest.py --manifest manifest.json --recreate`."
            )
        model = load_model()
        for rec in selected:
            query_modality = rec["query_modality"]
            target_modality = rec["target_modality"]
            pair = (query_modality, target_modality)
            flt = FilterBuilder().must(Field("modality").eq(target_modality)).build()
            hits = client.points.search(
                args.collection,
                embed_for(model, rec).tolist(),
                limit=args.k,
                filter=flt,
                with_payload=True,
            )
            relevant = set(rec["relevant"])
            found = any((hit.payload or {}).get("asset_id") in relevant for hit in hits)
            tally[pair][0] += int(found)
            tally[pair][1] += 1
            if found:
                continue

            misses += 1
            print(
                f"\nFAIL {query_modality} -> {target_modality} "
                f"hit@{args.k}: query={rec['query']!r} "
                f"expected asset_id in {sorted(relevant)}"
            )
            if not hits:
                print("  no results returned")
            else:
                for rank, hit in enumerate(hits, start=1):
                    print(f"  {rank:>2}. {hit.score:.4f}  {describe_hit(hit)}")

    print(f"\nHard-case failures: {misses} of {len(selected)} queries")
    for (query_modality, target_modality), (found, total) in sorted(tally.items()):
        print(
            f"  {query_modality:>5} -> {target_modality:<12} "
            f"hit@{args.k} {found}/{total}; failures {total - found}"
        )
    if not misses:
        print("No retrieval failures found for the selected queries.")


if __name__ == "__main__":
    main()
