"""Local embedding model. Vectors are L2-normalized, so the cosine score VectorAI DB returns equals the dot product.

Two backends:
  ollama                 an Ollama server (default http://localhost:11434); no PyTorch needed
  sentence-transformers  a Hugging Face model loaded in-process (install with `uv sync --extra st`)
"""
import hashlib
from pathlib import Path

import numpy as np
import requests

DEFAULT_MODEL = "embeddinggemma"   # Ollama's EmbeddingGemma; pull it with `ollama pull embeddinggemma`


def _ollama(texts: list[str], model: str, url: str, batch_size: int) -> np.ndarray:
    out = []
    for start in range(0, len(texts), batch_size):
        r = requests.post(f"{url}/api/embed", json={"model": model, "input": texts[start:start + batch_size]},
                          timeout=600)
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
        print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.asarray(out, dtype=np.float32)


def _sentence_transformers(texts: list[str], model: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer   # optional extra; slow import

    return SentenceTransformer(model).encode(texts, batch_size=batch_size, show_progress_bar=True,
                                             convert_to_numpy=True).astype(np.float32)


def embed(texts: list[str], model_name: str, cache_dir: Path, backend: str = "ollama",
          ollama_url: str = "http://localhost:11434", batch_size: int = 64) -> np.ndarray:
    """Embed texts, caching the result on disk keyed by backend, model, and content."""
    key = hashlib.sha256("\x00".join([backend, model_name, *texts]).encode()).hexdigest()[:16]
    cache = cache_dir / f"embeddings-{key}.npy"
    if cache.exists():
        return np.load(cache)

    if backend == "ollama":
        vectors = _ollama(texts, model_name, ollama_url, batch_size)
    else:
        vectors = _sentence_transformers(texts, model_name, batch_size)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, vectors)
    return vectors
