"""Shared helpers for the semantic-cache false-positive tests (tests 4 to 7).

Correctness definition used everywhere:
  - The cache holds Banking77 *train* questions. Each cached entry's answer is
    assumed to be the answer for its intent label.
  - A lookup is a HIT when the top-1 similarity score >= threshold.
  - A hit is WRONG (a false positive) when serving that cached answer would be
    incorrect for the incoming question (wrong intent, wrong tenant, or stale).
  - False-positive rate = wrong hits / hits  (the share of hits that are wrong).
"""

import argparse
import atexit
import hashlib
import csv
import json
import os
import platform
import random
import re
import urllib.request
from pathlib import Path

import numpy as np
from actian_vectorai import (
    Distance,
    Field,
    FilterBuilder,
    PointStruct,
    VectorAIClient,
    VectorParams,
)

SERVER = os.environ.get("VECTORAI_URL", "localhost:6574")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SEED = 42

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

BANKING77 = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}

THRESHOLD_GRID = np.round(np.arange(0.50, 0.995, 0.01), 2)


# ---------------------------------------------------------------- arguments
def base_args(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--threshold", type=float, default=0.80,
                   help="Operating similarity threshold (use the same one as test 1).")
    p.add_argument("--fp-target", type=float, default=0.05,
                   help="Correctness target: maximum acceptable false-positive rate.")
    p.add_argument("--train-per-intent", type=int, default=40,
                   help="Cached questions per intent (77 intents). 0 = all 10,003.")
    p.add_argument("--test-per-intent", type=int, default=10,
                   help="Lookup questions per intent (77 intents). 0 = all 3,080.")
    return p


def subset(rows, per_intent, seed):
    """Same number of questions from every intent, picked at random (seeded)."""
    if not per_intent:
        return rows
    by_intent = {}
    for r in rows:
        by_intent.setdefault(r["intent"], []).append(r)
    rng = random.Random(seed)
    out = []
    for intent in sorted(by_intent):
        group = by_intent[intent][:]
        rng.shuffle(group)
        out.extend(group[:per_intent])
    return out


# ---------------------------------------------------------------- data
def load_banking77(args=None):
    """Return (train, test) as lists of {"text", "intent"} dicts, subset per args."""
    out = {}
    for split, url in BANKING77.items():
        path = DATA / f"banking77_{split}.csv"
        if not path.exists():
            print(f"Downloading Banking77 {split} split...")
            urllib.request.urlretrieve(url, path)
        with open(path, newline="", encoding="utf-8") as f:
            out[split] = [{"text": r["text"].strip(), "intent": r["category"].strip()}
                          for r in csv.DictReader(f)]
    if args is not None:
        out["train"] = subset(out["train"], args.train_per_intent, SEED)
        out["test"] = subset(out["test"], args.test_per_intent, SEED + 1)
    print(f"Banking77 subset: {len(out['train'])} cached, {len(out['test'])} lookups, "
          f"{len({r['intent'] for r in out['train']})} intents")
    return out["train"], out["test"]


_model = None


def embed(texts, cache_name=None) -> np.ndarray:
    """Embed texts with normalized vectors; caches to data/ so reruns are fast."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL)
    digest = hashlib.md5("\n".join(texts).encode()).hexdigest()[:10]  # one file per subset
    path = DATA / f"emb_{slug}_{cache_name}_{digest}.npy" if cache_name else None
    if path and path.exists():
        vecs = np.load(path)
        if len(vecs) == len(texts):
            return vecs
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    vecs = _model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                         show_progress_bar=True).astype(np.float32)
    if path:
        np.save(path, vecs)
    return vecs


def assign_tenants(n, tenants=("acme", "globex", "initech"), seed=SEED):
    rng = random.Random(seed)
    return [rng.choice(tenants) for _ in range(n)]


# ---------------------------------------------------------------- VectorAI DB
def connect() -> VectorAIClient:
    """Open a connected client and close it when the script exits.

    Same as `with VectorAIClient(SERVER) as client:` in Actian's docs, without
    indenting every test under a with-block.
    """
    client = VectorAIClient(SERVER)
    client.__enter__()
    atexit.register(client.__exit__, None, None, None)
    return client


COMMUNITY_VECTOR_LIMIT = 5000  # VectorAI DB Community Edition, across all collections
TEST_COLLECTIONS = [f"fp_test{n}" for n in range(1, 8)]


def build_collection(client, name, vectors, payloads, batch=256):
    """(Re)create a cosine collection and load every cached entry into it.

    Community Edition allows 5,000 vectors in total, so every earlier test
    collection is deleted first. Each test only needs its own collection while
    it runs; its results are already saved in results/.
    """
    if len(vectors) > COMMUNITY_VECTOR_LIMIT:
        raise SystemExit(
            f"This run needs {len(vectors)} vectors, above the Community Edition limit of "
            f"{COMMUNITY_VECTOR_LIMIT}. Lower --train-per-intent (40 per intent = 3,080 vectors).")
    for old in TEST_COLLECTIONS:
        try:
            client.collections.delete(old)
            print(f"Deleted old collection '{old}'")
        except Exception as e:
            if "not found" not in str(e).lower() and "not exist" not in str(e).lower():
                print(f"Could not delete '{old}': {type(e).__name__}: {e}")
    client.collections.get_or_create(
        name=name,
        vectors_config=VectorParams(size=int(vectors.shape[1]), distance=Distance.Cosine),
    )
    for start in range(0, len(vectors), batch):
        points = [PointStruct(id=i, vector=vectors[i].tolist(), payload=payloads[i])
                  for i in range(start, min(start + batch, len(vectors)))]
        client.points.upsert(name, points=points)
    client.vde.flush(name)
    print(f"Collection '{name}': {len(vectors)} cached entries loaded.")


def top1(client, collection, vector, flt=None):
    """Return (score, payload) of the nearest cached entry, or (nan, None)."""
    res = client.points.search(collection, vector=vector.tolist(), limit=1,
                               with_payload=True, filter=flt) or []
    if not res:
        return float("nan"), None
    return float(res[0].score), res[0].payload


def lookup_all(client, collection, vectors, filter_fn=None, label=""):
    """Run top-1 lookups for every query. filter_fn(i) -> filter or None."""
    scores, payloads = [], []
    for i, v in enumerate(vectors):
        s, p = top1(client, collection, v, filter_fn(i) if filter_fn else None)
        scores.append(s)
        payloads.append(p)
        if (i + 1) % 500 == 0:
            print(f"  {label} {i + 1}/{len(vectors)} lookups")
    return np.array(scores, dtype=np.float64), payloads


# ---------------------------------------------------------------- metrics
def rates(scores, correct, threshold, mask=None):
    """Hit rate and false-positive rate at one global threshold."""
    scores = np.asarray(scores)
    correct = np.asarray(correct, dtype=bool)
    mask = np.ones(len(scores), bool) if mask is None else np.asarray(mask, bool)
    hit = mask & (np.nan_to_num(scores, nan=-1.0) >= threshold)
    return summarize(hit, correct, mask, threshold)


def summarize(hit, correct, mask, threshold=None):
    n = int(mask.sum())
    h = int(hit.sum())
    wrong = int((hit & ~correct).sum())
    return {
        "threshold": None if threshold is None else float(threshold),
        "queries": n,
        "hits": h,
        "hit_rate": h / n if n else 0.0,
        "wrong_hits": wrong,
        "fp_rate": wrong / h if h else 0.0,
        "wrong_hits_per_100_queries": 100 * wrong / n if n else 0.0,
    }


def sweep(scores, correct, mask=None):
    return [rates(scores, correct, t, mask) for t in THRESHOLD_GRID]


def threshold_for_target(scores, correct, target, mask=None, min_hits=1):
    """Lowest threshold whose false-positive rate meets the target."""
    for r in sweep(scores, correct, mask):
        if r["hits"] >= min_hits and r["fp_rate"] <= target:
            return r["threshold"]
    return None


def split_halves(n, seed=SEED):
    """Boolean masks for a calibration half and an evaluation half."""
    idx = np.random.default_rng(seed).permutation(n)
    cal = np.zeros(n, bool)
    cal[idx[: n // 2]] = True
    return cal, ~cal


def pct(x):
    return f"{100 * x:.1f}%"


def hardware():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return f"Apple MPS ({platform.machine()})"
    except Exception:
        pass
    return f"CPU ({platform.processor() or platform.machine()})"


def save(name, payload):
    payload["embedding_model"] = EMBED_MODEL
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {path}")


__all__ = [n for n in dir() if not n.startswith("_")] + ["Field", "FilterBuilder"]
