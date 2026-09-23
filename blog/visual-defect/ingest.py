# ingest.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

from actian_vectorai import (
    VectorAIClient, PointStruct, VectorParams, Distance, CollectionExistsError,
)

from clip_encoder import ClipEncoder, EMBED_DIM

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def scan_corpus(root: Path) -> list[dict]:
    """Read metadata straight out of the directory layout."""
    records: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "ground_truth" in path.parts:      # segmentation masks, not photos
            continue
        try:
            rel = path.relative_to(root)
            part_category, _split, defect_type = rel.parts[0], rel.parts[1], rel.parts[2]
        except (ValueError, IndexError):
            continue
        records.append(
            {
                "image_path": str(path),
                "part_category": part_category,
                "defect_type": defect_type,
                "is_defect": defect_type != "good",
            }
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./mvtec_ad")
    ap.add_argument("--collection", default="defect_search")
    ap.add_argument("--url", default="localhost:6574")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    data_root = Path(args.data)
    if not data_root.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {data_root}\n"
            "Download and extract MVTec AD, then pass its directory with --data."
        )

    records = scan_corpus(data_root)
    if not records:
        raise SystemExit(
            f"No dataset images found under {data_root}\n"
            "Expected paths like <data>/<category>/test/<defect>/*.png."
        )
    print(f"Found {len(records)} images")

    encoder = ClipEncoder()
    print(f"CLIP running on {encoder.device}")

    with VectorAIClient(args.url) as client:
        try:
            client.collections.create(
                args.collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.Cosine),
            )
            print(f"Created collection '{args.collection}'")
        except CollectionExistsError:
            print(f"Collection '{args.collection}' already exists, appending")

        started = time.perf_counter()
        done = 0

        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            vectors = encoder.encode_images([r["image_path"] for r in batch])

            client.points.upsert(
                args.collection,
                points=[
                    PointStruct(id=start + offset, vector=vec.tolist(), payload=record)
                    for offset, (record, vec) in enumerate(zip(batch, vectors))
                ],
            )

            done += len(batch)
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0.0
            remaining = (len(records) - done) / rate if rate else 0.0
            print(
                f"\r{done}/{len(records)} images "
                f"| {rate:.1f} img/s | ~{remaining/60:.1f} min left",
                end="", flush=True,
            )

        print(f"\nIndexed {done} images in {(time.perf_counter()-started)/60:.1f} min")
        print(f"Collection now holds {client.points.count(args.collection)} points")


if __name__ == "__main__":
    main()