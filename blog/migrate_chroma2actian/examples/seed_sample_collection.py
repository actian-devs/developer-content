#!/usr/bin/env python3
"""Write a small, realistic test collection into Chroma (local, self-hosted server, or Chroma Cloud).

Uses synthetic vectors (no embedding model or network needed): 5 topic clusters, so nearest-neighbour
results are meaningful and the migration's search-parity check has something real to compare.

    python examples/seed_sample_collection.py --source local --chroma-path ./chroma_demo
    python examples/seed_sample_collection.py --source http  --chroma-host localhost --chroma-port 8000
    CHROMA_API_KEY=.. CHROMA_TENANT=.. CHROMA_DATABASE=.. python examples/seed_sample_collection.py --source cloud

Environment fallbacks: CHROMA_PATH, CHROMA_HOST, CHROMA_PORT, CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE.
"""
from __future__ import annotations

import argparse
import os

import chromadb
import numpy as np

TOPICS = ["billing", "shipping", "returns", "accounts", "security"]


def client_from(a: argparse.Namespace):
    if a.source == "local":
        return chromadb.PersistentClient(path=a.chroma_path)
    if a.source == "http":
        return chromadb.HttpClient(host=a.chroma_host, port=a.chroma_port)
    return chromadb.CloudClient(tenant=a.chroma_tenant, database=a.chroma_database, api_key=a.chroma_api_key)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=("local", "http", "cloud"), required=True)
    p.add_argument("--chroma-path", default=os.environ.get("CHROMA_PATH", "./chroma_demo"))
    p.add_argument("--chroma-host", default=os.environ.get("CHROMA_HOST", "localhost"))
    p.add_argument("--chroma-port", type=int, default=int(os.environ.get("CHROMA_PORT", "8000")))
    p.add_argument("--chroma-api-key", default=os.environ.get("CHROMA_API_KEY"))
    p.add_argument("--chroma-tenant", default=os.environ.get("CHROMA_TENANT"))
    p.add_argument("--chroma-database", default=os.environ.get("CHROMA_DATABASE"))
    p.add_argument("--collection", default="support_tickets_demo")
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--space", choices=("cosine", "l2", "ip"), default="cosine")
    p.add_argument("--recreate", action="store_true", help="delete the collection first if it exists")
    a = p.parse_args()

    client = client_from(a)
    if a.recreate:
        try:
            client.delete_collection(a.collection)
        except Exception:  # noqa: BLE001 - not there yet
            pass
    col = client.create_collection(a.collection, configuration={"hnsw": {"space": a.space}})

    rng = np.random.default_rng(42)
    centers = rng.normal(size=(len(TOPICS), a.dim))
    for start in range(0, a.count, 100):
        n = min(100, a.count - start)
        idx = list(range(start, start + n))
        topic_ids = [i % len(TOPICS) for i in idx]
        vecs = np.stack([centers[t] + 0.35 * rng.normal(size=a.dim) for t in topic_ids]).astype(np.float32)
        col.add(
            ids=[f"ticket-{i:05d}" for i in idx],
            embeddings=vecs.tolist(),
            documents=[f"Support ticket {i} about {TOPICS[t]}." for i, t in zip(idx, topic_ids)],
            metadatas=[{
                "topic": TOPICS[t], "priority": i % 4, "resolved": i % 3 == 0, "csat": round(1 + (i % 50) / 12.5, 2),
                "tags": [TOPICS[t], f"p{i % 4}"],
                # reserved-name clash on purpose: the migration renames it to meta_document and reports it
                **({"document": f"legacy-ref-{i}"} if i % 100 == 0 else {}),
            } for i, t in zip(idx, topic_ids)],
        )
    print(f"Wrote {col.count()} records to collection {a.collection!r} ({a.space}, dim {a.dim}).")


if __name__ == "__main__":
    main()
