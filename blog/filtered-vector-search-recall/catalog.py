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