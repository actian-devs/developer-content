# Run guide: filtered vector search recall test

This guide builds the whole project from an empty folder. You create every file yourself, run each step, and check the output before you move on. Nothing is cloned.

## Step 0. What you need

- `uv` ([install instructions](https://docs.astral.sh/uv/getting-started/installation/)). `uv` installs Python 3.12 for you.
- Docker, running. VectorAI DB runs in a container.
- An internet connection for the first run, because `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (about 90 MB) and `uv` downloads PyTorch (about 2 GB on Linux and Windows).
- A terminal. Every command below runs in the project folder.

## Step 1. Create the project

```bash
mkdir filtered-vector-search-recall
cd filtered-vector-search-recall
uv init --bare
uv python pin 3.12
uv add numpy "sentence-transformers==6.1.0" "actian-vectorai-client==1.0.3"
uv add --optional plot matplotlib
uv sync --extra plot
```

Expected result. The folder holds `pyproject.toml`, `.python-version`, `uv.lock`, and a `.venv` folder. `uv sync` ends without an error.

Check the SDK imports:

```bash
uv run python -c "import actian_vectorai, numpy, sentence_transformers; print('imports ok')"
```

## Step 2. Start VectorAI DB

Create `docker-compose.yml`:

```yaml
services:
  vectorai:
    image: actian/vectorai:latest
    container_name: vectorai
    ports:
      - "6573:6573"   # REST API
      - "6574:6574"   # gRPC API, used by the Python SDK
      - "6575:6575"   # local web UI
    volumes:
      - ./vectorai_data:/var/lib/actian-vectorai
    environment:
      - ACTIAN_VECTORAI_ACCEPT_EULA=YES
    restart: unless-stopped
```

Start the container and check it:

```bash
docker compose up -d
docker compose ps
```

Expected result. `docker compose ps` lists `vectorai` with status `running` and ports 6573, 6574, and 6575. The SDK talks to port 6574. If a port is already in use, change the left-hand number in `docker-compose.yml` and pass `--url localhost:<port>` to `load.py`, `measure.py`, and `clauses.py`.

Actian publishes no image tag list, so record the digest you pulled and put it in your results notes:

```bash
docker image inspect actian/vectorai:latest --format "{{index .RepoDigests 0}}"
```

## Step 3. Build and embed the catalog

Create `catalog.py`. It generates 5,000 products and 100 queries with a fixed seed and embeds them:

```python
"""Generate a seeded product catalog and embed it with sentence-transformers.

Writes catalog.npz (vectors and query vectors) and catalog.json (payloads, query text, model name).
Price is drawn independently of category and text, so a price ceiling changes how many products
a filter admits without changing what the filter means. Filters that correlate with the query
behave differently, so rerun the test with your own filters before you trust any number.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

CATEGORIES = {
    "electronics": ["wireless earbuds", "noise cancelling headphones", "smartwatch", "bluetooth speaker",
                    "portable charger", "gaming mouse", "mechanical keyboard", "webcam", "tablet",
                    "laptop stand", "action camera", "e-reader"],
    "home_kitchen": ["air fryer", "chef knife", "coffee grinder", "blender", "cast iron skillet",
                     "electric kettle", "food storage set", "rice cooker", "vacuum cleaner",
                     "robot mop", "toaster oven", "dish rack"],
    "sports_outdoors": ["trail running shoes", "yoga mat", "camping tent", "hydration pack", "bike helmet",
                        "resistance bands", "sleeping bag", "trekking poles", "jump rope", "water bottle",
                        "climbing harness", "fishing reel"],
    "beauty": ["face moisturizer", "vitamin c serum", "hair dryer", "electric toothbrush", "sunscreen",
               "beard trimmer", "shampoo", "lip balm", "makeup brush set", "nail kit", "body lotion",
               "facial roller"],
    "toys_games": ["building blocks", "board game", "puzzle set", "remote control car", "plush bear",
                   "art supply kit", "card game", "toy kitchen", "wooden train", "science kit",
                   "dollhouse", "kite"],
    "books": ["cookbook", "mystery novel", "travel guide", "history book", "science fiction paperback",
              "children's picture book", "biography", "language workbook", "poetry collection",
              "graphic novel", "gardening handbook", "study guide"],
    "clothing": ["rain jacket", "wool socks", "denim jeans", "running shorts", "winter gloves",
                 "cotton t-shirt", "leather belt", "fleece hoodie", "sun hat", "hiking boots",
                 "silk scarf", "work apron"],
    "garden": ["raised garden bed", "pruning shears", "garden hose", "solar path lights", "compost bin",
               "seed starter kit", "watering can", "lawn sprinkler", "plant stand", "bird feeder",
               "weed barrier", "garden kneeler"],
    "pet_supplies": ["dog harness", "cat scratching post", "automatic feeder", "pet carrier", "chew toy",
                     "aquarium filter", "dog bed", "litter box", "grooming brush", "pet water fountain",
                     "bird cage", "leash"],
    "office": ["standing desk", "ergonomic chair", "desk lamp", "monitor arm", "notebook set",
               "label printer", "document scanner", "whiteboard", "cable organizer", "paper shredder",
               "wireless presenter", "filing cabinet"],
}
ADJECTIVES = ["compact", "durable", "lightweight", "premium", "budget", "waterproof", "ergonomic",
              "portable", "rechargeable", "foldable", "quiet", "adjustable", "insulated", "stackable",
              "non-slip", "energy efficient", "heavy duty", "travel size", "eco friendly", "easy clean"]
USES = ["daily use", "travel", "small spaces", "gifts", "beginners", "professionals", "the whole family",
        "long trips", "everyday comfort", "busy mornings"]
BRANDS = ["Norvik", "Aldera", "Brightfern", "Calder and Finch", "Dunmore", "Everly", "Fenwick", "Galloway Lane",
          "Halden", "Ironvale", "Juniper Row", "Kestrel Hill", "Lumeo", "Marlow", "Northgate", "Oakmere",
          "Pelham", "Quillon", "Ravenscroft", "Sorrel"]
FEATURES = ["Built to last through years of regular use.", "Designed for a simple setup with no tools.",
            "Comes with a one year limited warranty.", "Tested for reliability in everyday conditions.",
            "Packs down small for storage.", "A favorite with reviewers for value."]


def generate_products(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    prices = np.clip(rng.lognormal(mean=np.log(40), sigma=0.8, size=n), 3, 600).round(2)
    ratings = np.clip(rng.normal(4.0, 0.6, size=n), 1.0, 5.0).round(1)
    conditions = rng.choice(["new", "refurbished", "used"], size=n, p=[0.8, 0.12, 0.08])
    in_stock = rng.random(n) < 0.75
    products = []
    for i in range(n):
        cat = cats[int(rng.integers(len(cats)))]
        noun = CATEGORIES[cat][int(rng.integers(len(CATEGORIES[cat])))]
        adj = ADJECTIVES[int(rng.integers(len(ADJECTIVES)))]
        brand = BRANDS[int(rng.integers(len(BRANDS)))]
        use = USES[int(rng.integers(len(USES)))]
        feature = FEATURES[int(rng.integers(len(FEATURES)))]
        products.append({
            "idx": i,
            "title": f"{brand} {adj} {noun}",
            "description": f"{'An' if adj[0] in 'aeiou' else 'A'} {adj} {noun} made for {use}. {feature}",
            "category": cat,
            "price": float(prices[i]),
            "rating": float(ratings[i]),
            "condition": str(conditions[i]),
            "in_stock": bool(in_stock[i]),
        })
    return products


def generate_queries(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed + 1)
    cats = list(CATEGORIES)
    templates = ["{adj} {noun} for {use}", "best {noun} for {use}", "looking for {adj} {noun}",
                 "affordable {noun} for {use}"]
    queries = []
    for i in range(n):
        cat = cats[i % len(cats)]                      # every category gets the same number of queries
        noun = CATEGORIES[cat][int(rng.integers(len(CATEGORIES[cat])))]
        text = templates[int(rng.integers(len(templates)))]
        queries.append(text.format(adj=ADJECTIVES[int(rng.integers(len(ADJECTIVES)))], noun=noun,
                                   use=USES[int(rng.integers(len(USES)))]))
    return queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default="catalog")
    args = ap.parse_args()

    products = generate_products(args.n, args.seed)
    queries = generate_queries(args.queries, args.seed)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    texts = [f"{p['title']}. {p['description']}" for p in products]
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    qvecs = model.encode(queries, normalize_embeddings=True)

    np.savez(f"{args.out}.npz", vectors=vecs.astype("float32"), queries=qvecs.astype("float32"))
    with open(f"{args.out}.json", "w") as f:
        json.dump({"model": args.model, "seed": args.seed, "products": products, "queries": queries}, f)
    print(f"wrote {len(products)} products and {len(queries)} queries "
          f"({vecs.shape[1]}-dimensional) to {args.out}.npz and {args.out}.json")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run catalog.py
```

Expected result. A progress bar for the embedding step, then one line.

```text
wrote 5000 products and 100 queries (384-dimensional) to catalog.npz and catalog.json
```

On a CPU this takes a few minutes. The folder now holds `catalog.npz` (vectors) and `catalog.json` (payloads and query text). The same seed gives the same catalog on every run.

## Step 4. Load the catalog into VectorAI DB

Create `common.py`. Every later script imports it:

```python
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
```

Create `load.py`:

```python
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
```

Run it:

```bash
uv run load.py
```

Expected result.

```text
loaded 5000 products (384-dimensional, sentence-transformers/all-MiniLM-L6-v2) into 'products'
```

If this fails with a connection error, the container is not running or the port differs. Re-run `docker compose ps`.

## Step 5. Measure recall

Create `measure.py`:

```python
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
```

Run it. Pass your own hardware description, because the output header records it:

```bash
uv run measure.py --hardware "<CPU, RAM, GPU>" --hnsw-ef 64 256 1024 --overfetch 5 20 --probe-limits 10 50 200 1000
```

Expected result. The header line, then one block per selectivity. Your numbers will differ from any example, so only the shape matters here.

```text
Hardware: <your hardware> | actian-vectorai-client 1.0.3 | embeddings: sentence-transformers/all-MiniLM-L6-v2 | 5000 products, 100 queries | k=10

filter price <= <ceiling>: admits 2500 products (50.0%); the server counts 2500
  post_filter              recall@10 <r>   returned <n>   off-filter 0
  vectorai_filter          recall@10 <r>   returned <n>   off-filter 0
  vectorai_ef64            recall@10 <r>   returned <n>   off-filter 0
  ...
  limit probe: limit=10 returned <n> filtered results (the filter admits 2500)
  ...
```

Check three things before you copy any number.

1. The line that starts with `filter price` must end with the same count on both sides. A `MISMATCH` warning means the collection and the NumPy mask disagree, so reload with `uv run load.py` and run again.
2. Every `off-filter` value must read `0`. A non-zero value means the server returned products that break the filter, which is a correctness problem and belongs in the article.
3. The `post_filter` row must show `recall` equal to `returned` divided by 10. If it does not, something is wrong with the run.

The script writes `recall.csv`, and `probe.csv` because of `--probe-limits`.

## Step 6. Plot the result

Create `plot.py`:

```python
"""Plot recall at 10 by filter selectivity from recall.csv (written by measure.py)."""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
# categorical slots 1 to 4 of the reference palette, assigned in fixed order
SERIES = ["post_filter", "vectorai_filter"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LABELS = {"post_filter": "Post-filter", "vectorai_filter": "VectorAI DB filter"}


def label(method: str) -> str:
    if method in LABELS:
        return LABELS[method]
    if method.startswith("vectorai_ef"):
        return f"VectorAI DB, hnsw_ef={method[len('vectorai_ef'):]}"
    return f"VectorAI DB, {method[len('vectorai_overfetch'):]}x overfetch"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="recall.csv")
    ap.add_argument("--out", default="images/recall-by-selectivity.png")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    shares = sorted({float(r["share"]) for r in rows}, reverse=True)
    data = defaultdict(dict)
    for r in rows:
        data[r["method"]][float(r["share"])] = float(r["recall_at_k"])
    # Variants whose curve equals the plain filter's curve add nothing to the picture, so leave them out.
    base = [data["vectorai_filter"].get(s) for s in shares]
    extra = [m for m in data if m not in SERIES and [data[m].get(s) for s in shares] != base]
    extra.sort(key=lambda m: sum(data[m].values()))          # weaker variant first, so colors stay stable
    methods = SERIES + extra[:2]                              # four series at most
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = list(range(len(shares)))
    ends = []
    for m, c in zip(methods, COLORS):
        ys = [data[m].get(s, float("nan")) for s in shares]
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=label(m), zorder=3)
        ends.append([ys[-1], label(m), c])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):                             # nudge end labels apart so they never overlap
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.07)
    for y, text, c in ends:
        ax.annotate(text, (xs[-1], y), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_xticks(xs, [f"{s:.0%}" for s in shares], color=MUTED)
    ax.set_xlim(-0.2, len(xs) - 1 + 1.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1.0"], color=MUTED)
    ax.set_xlabel("Share of the catalog the filter admits", color=MUTED)
    ax.set_ylabel("Recall at 10", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run plot.py
```

Expected result. `wrote images/recall-by-selectivity.png`.

## Step 7. Run the must, should, and must-not filters

Create `clauses.py`:

```python
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
```

Run it:

```bash
uv run clauses.py
```

Expected result. Two blocks of up to five products each. Every line starts with `ok`, and a line that starts with `BAD` means a returned product broke the rule. A `recall loss` line means the filter admits more products than the search returned.

## Step 8. Fill the article placeholders

| Article placeholder | Where it comes from |
|---|---|
| Table, post-filter recall at 10 | `recall_at_k` of `post_filter` in `recall.csv` |
| Table, VectorAI DB recall at 10 | `recall_at_k` of `vectorai_filter` |
| Table, VectorAI DB results returned | `returned` of `vectorai_filter` |
| Off-filter confirmation | the `off_filter` column, zero in every row |
| `hnsw_ef=256` recall at 1% | `recall_at_k` of `vectorai_ef256` at share 0.01 |
| 20-times overfetch recall at 1% | `recall_at_k` of `vectorai_overfetch20` at share 0.01 |
| Hardware and image digest | your `--hardware` string and the digest from Step 2 |

Read `probe.csv` to decide what to write about the candidate window. If `returned` grows with `limit`, overfetching adds results. If it stays flat, only `hnsw_ef` can help.

## Troubleshooting

- **`UnimplementedError` on count.** `count_matching` already falls back to `points.scroll`, so you should see no error.
- **`vectorai_filter` returns fewer than 10 results.** This is the behavior the tutorial measures. Report it and run the `hnsw_ef` and overfetch variants.
- **The embedding model fails to download.** Check your internet connection and the Hugging Face status, then run `uv run catalog.py` again.
- **`ModuleNotFoundError: matplotlib`.** Run `uv sync --extra plot`.
- **A search returns an empty list right after loading.** Run `uv run load.py` again. It waits for the index before it exits.