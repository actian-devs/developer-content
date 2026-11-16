from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_actian import FakeVectorAI  # noqa: E402

from chroma2actian.config import ActianConfig, ChromaConfig  # noqa: E402
from chroma2actian.source import ChromaSource  # noqa: E402
from chroma2actian.target import ActianTarget  # noqa: E402
from chroma2actian.transform import TransformOptions  # noqa: E402


@pytest.fixture
def chroma_path(tmp_path: Path) -> Path:
    p = tmp_path / "chroma"
    p.mkdir()
    return p


@pytest.fixture
def make_collection(chroma_path: Path):
    """Create a Chroma collection with deterministic vectors/metadata; returns its name."""

    def _make(name: str = "docs", n: int = 237, dim: int = 16, space: str = "cosine", *, ids: list[str] | None = None,
              metadata_fn=None, documents: bool = True, seed: int = 7, vectors: np.ndarray | None = None) -> str:
        client = chromadb.PersistentClient(path=str(chroma_path))
        col = client.create_collection(name, configuration={"hnsw": {"space": space}})
        rng = np.random.default_rng(seed)
        vecs = vectors if vectors is not None else rng.normal(size=(n, dim)).astype(np.float32)
        ids = ids or [f"doc-{i}" for i in range(len(vecs))]
        for s in range(0, len(ids), 100):
            sl = slice(s, s + 100)
            mds = [metadata_fn(i) if metadata_fn else {"idx": i, "grp": f"g{i % 5}", "score": i / 3.0, "flag": i % 2 == 0}
                   for i in range(s, min(s + 100, len(ids)))]
            col.add(ids=ids[sl], embeddings=vecs[sl].tolist(), metadatas=mds,
                    documents=[f"text {i}" for i in range(s, min(s + 100, len(ids)))] if documents else None)
        return name

    return _make


@pytest.fixture
def source(chroma_path: Path) -> ChromaSource:
    s = ChromaSource(ChromaConfig(mode="local", path=str(chroma_path), default_page_size=50), sleep=lambda _s: None)
    s.connect()
    return s


@pytest.fixture
def fake() -> FakeVectorAI:
    return FakeVectorAI()


@pytest.fixture
def target(fake: FakeVectorAI) -> ActianTarget:
    return ActianTarget(ActianConfig(url="fake:6574"), client=fake, sleep=lambda _s: None)


@pytest.fixture
def topts() -> TransformOptions:
    return TransformOptions()
