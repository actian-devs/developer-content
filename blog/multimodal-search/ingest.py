from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from actian_vectorai import (
    CollectionExistsError, Distance, PointStruct, VectorAIClient, VectorParams,
)

from embed import DIM, embed_document, embed_media, load_model

MEDIA_KIND = {"manual_page": "image", "photo": "image", "video": "video", "voice": "audio"}
MAX_AUDIO_SECONDS = 327   # 8K shared context at 25 audio tokens per second (model card)
MAX_WORDS_PER_CHUNK = 300  # keeps a text chunk far below the 8,192-token limit


def chunk_words(text: str, size: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)] or [""]


def check_audio_length(path: str) -> None:
    seconds = sf.info(path).duration
    if seconds > MAX_AUDIO_SECONDS:
        raise SystemExit(f"{path} runs {seconds:.0f}s, over the {MAX_AUDIO_SECONDS}s limit. Split it first.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--collection", default="field_search")
    ap.add_argument("--url", default="localhost:6584")
    ap.add_argument("--recreate", action="store_true", help="drop the collection first")
    ap.add_argument("--from-cache", action="store_true",
                    help="reload embeddings.npy + meta.json after verifying they match --manifest")
    args = ap.parse_args()

    if args.from_cache:
        vectors_array = np.load("embeddings.npy")
        rows = json.loads(Path("meta.json").read_text())
        records = json.loads(Path(args.manifest).read_text())
        expected = Counter()
        for record in records:
            modality = record["modality"]
            expected[modality] += (
                len(chunk_words(record["text"])) if modality == "manual_text" else 1
            )
        cached = Counter(row.get("modality") for row in rows)
        if (
            vectors_array.ndim != 2
            or len(vectors_array) != len(rows)
            or cached != expected
        ):
            raise SystemExit(
                f"Cached embeddings do not match {args.manifest}: "
                f"expected {sum(expected.values())} vectors with modalities "
                f"{dict(expected)}, found {len(rows)} metadata rows with modalities "
                f"{dict(cached)}. Nothing was uploaded. Rebuild with "
                f"`uv run ingest.py --manifest {args.manifest} --recreate`."
            )
        vectors = list(vectors_array)
        print(f"loaded {len(vectors)} cached vectors")
    else:
        rows, vectors = embed_manifest(args.manifest)
        np.save("embeddings.npy", np.stack(vectors))
        Path("meta.json").write_text(json.dumps(rows))

    upload(args, rows, vectors)


def embed_manifest(manifest: str) -> tuple[list[dict], list[np.ndarray]]:
    records = json.loads(Path(manifest).read_text())
    model = load_model()
    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    for n_rec, rec in enumerate(records, start=1):
        print(f"\rembedding {n_rec}/{len(records)} ({rec['modality']})", end="", flush=True)
        if rec["modality"] == "manual_text":
            for n, chunk in enumerate(chunk_words(rec["text"])):
                rows.append({"asset_id": rec["asset_id"], "modality": "manual_text", "chunk": n,
                             "text": chunk, "source": rec.get("source", "")})
                vectors.append(embed_document(model, chunk))
            continue
        kind = MEDIA_KIND[rec["modality"]]
        if kind == "audio":
            check_audio_length(rec["path"])
        rows.append({"asset_id": rec["asset_id"], "modality": rec["modality"], "path": rec["path"]})
        vectors.append(embed_media(model, kind, rec["path"]))

    print(f"\n{len(vectors)} vectors embedded")
    return rows, vectors


def upload(args: argparse.Namespace, rows: list[dict], vectors: list[np.ndarray]) -> None:
    with VectorAIClient(args.url) as client:
        if args.recreate:
            try:
                client.collections.delete(args.collection)
                print(f"Dropped '{args.collection}'")
            except Exception:
                pass
        try:
            client.collections.create(
                args.collection, vectors_config=VectorParams(size=DIM, distance=Distance.Cosine)
            )
        except CollectionExistsError:
            print(f"Collection '{args.collection}' exists, appending")
        points = [PointStruct(id=i, vector=v.tolist(), payload=row)
                  for i, (row, v) in enumerate(zip(rows, vectors))]
        for start in range(0, len(points), 256):   # batches keep each gRPC message small
            client.points.upsert(args.collection, points=points[start : start + 256])
        print(f"{client.points.count(args.collection)} points in '{args.collection}'")


if __name__ == "__main__":
    main()