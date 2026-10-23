from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
DIM = 768  # native size; the model card also lists 512, 256 and 128


def load_model() -> SentenceTransformer:
    # The model card warns that float16 can return NaN, so use bf16 where the
    # GPU supports it and float32 everywhere else.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    return SentenceTransformer(MODEL_ID, model_kwargs={"torch_dtype": dtype})


def _check(vec: np.ndarray) -> np.ndarray:
    if not np.isfinite(vec).all():
        raise ValueError("Embedding contains NaN or inf. Check the dtype (use bf16 or fp32).")
    return vec


def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    return _check(model.encode(text, prompt_name="SearchQuery", normalize_embeddings=True))


def embed_document(model: SentenceTransformer, text: str) -> np.ndarray:
    return _check(model.encode(text, prompt_name="Document", normalize_embeddings=True))


def embed_media(model: SentenceTransformer, kind: str, path: str) -> np.ndarray:
    """kind is 'image', 'audio' or 'video'. Media takes no prompt_name."""
    if kind not in {"image", "audio", "video"}:
        raise ValueError(f"unknown media kind: {kind}")
    return _check(model.encode({kind: str(path)}, normalize_embeddings=True))