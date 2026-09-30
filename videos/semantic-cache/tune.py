"""Sweep the similarity threshold over labelled pairs and print where it holds."""

import json
import pathlib

from fastembed import TextEmbedding

from app.cache import EMBED_MODEL, THRESHOLD

PAIRS = pathlib.Path(__file__).parent / "data" / "labelled_pairs.jsonl"


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


def main():
    pairs = [json.loads(line) for line in PAIRS.read_text(encoding="utf-8").splitlines() if line.strip()]
    embed = TextEmbedding(EMBED_MODEL)

    scored = []
    for pair in pairs:
        a, b = (v.tolist() for v in embed.embed([pair["a"], pair["b"]]))
        scored.append((cosine(a, b), pair["same"]))

    same = sum(1 for _, s in scored if s)
    different = len(scored) - same

    print(f"{len(scored)} pairs | {same} same intent | {different} different\n")
    print("threshold   hit rate   false hits")
    for step in range(50, 100, 2):
        threshold = step / 100
        hits = sum(1 for score, s in scored if s and score >= threshold)
        false = sum(1 for score, s in scored if not s and score >= threshold)
        marker = "   <- THRESHOLD in app/cache.py" if abs(threshold - THRESHOLD) < 1e-9 else ""
        print(f"   {threshold:.2f}      {hits / same:6.0%}     {false}/{different}{marker}")


if __name__ == "__main__":
    main()
