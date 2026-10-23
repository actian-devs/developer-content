from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymupdf

from embed import load_model


def pdf_chunks(source: Path, words_per_chunk: int) -> list[dict[str, str | int]]:
    chunks = []
    for pdf in sorted(source.rglob("*.pdf")):
        with pymupdf.open(pdf) as document:
            for page_number, page in enumerate(document, start=1):
                words = page.get_text().split()
                for start in range(0, len(words), words_per_chunk):
                    text = " ".join(words[start : start + words_per_chunk]).strip()
                    if text:
                        chunks.append({
                            "asset_id": pdf.stem,
                            "modality": "manual_text",
                            "source": str(pdf),
                            "page": page_number,
                            "text": text,
                        })
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and embed distinct text chunks from PDF manuals."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True, help="number of vectors to prepare")
    parser.add_argument("--chunk-words", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Source directory not found: {args.source}")
    if args.target < 1 or args.chunk_words < 1 or args.batch_size < 1:
        raise SystemExit("target, chunk-words, and batch-size must all be positive")

    chunks = pdf_chunks(args.source, args.chunk_words)
    if not chunks:
        raise SystemExit(f"No extractable PDF text found under {args.source}")
    if len(chunks) < args.target:
        raise SystemExit(
            f"Found {len(chunks)} text chunks, fewer than target {args.target}. "
            "Add more manuals or lower --chunk-words/--target; chunks are not duplicated."
        )

    selected = chunks[: args.target]
    model = load_model()
    vectors = model.encode(
        [chunk["text"] for chunk in selected],
        prompt_name="Document",
        normalize_embeddings=True,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding contains NaN or inf")

    metadata = [{key: value for key, value in chunk.items() if key != "text"}
                for chunk in selected]
    np.save("filter_embeddings.npy", vectors)
    Path("filter_meta.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote {len(vectors)} distinct vectors to filter_embeddings.npy and filter_meta.json")


if __name__ == "__main__":
    main()