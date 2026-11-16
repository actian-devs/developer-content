"""Helpers shared by load.py, measure.py and clauses.py."""
from __future__ import annotations

import json
from importlib.metadata import version

import numpy as np

URL = "localhost:6574"
COLLECTION = "products"


def load_catalog(npz="catalog.npz", meta="catalog.json"):
    data = np.load(npz)
    info = json.load(open(meta))
    return data["vectors"], data["queries"], info["products"], info


def count_matching(client, collection: str, flt) -> int:
    """Count the points a filter admits.

    The predicate filters tutorial says points.count raises UnimplementedError in VectorAI DB 1.0.0,
    so try it first and fall back to paging through points.scroll.
    """
    try:
        return client.points.count(collection, filter=flt)
    except Exception:                                    # noqa: BLE001 - any server-side refusal
        total, offset = 0, None
        while True:
            page, offset = client.points.scroll(collection, limit=1000, filter=flt,
                                                with_payload=False, offset=offset)
            total += len(page)
            if offset is None or not page:
                return total


def versions() -> str:
    return f"actian-vectorai-client {version('actian-vectorai-client')}"