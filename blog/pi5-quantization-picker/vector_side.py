"""Shrinking the vector side: index memory and recall for HNSW against IVF with product quantization.

Builds the same 4,000-chunk catalog under each index type and records VectorAI DB's own
index_memory_bytes, the container's memory, recall@10 against an exact search, and how often the
right chunk reaches the top 3 for the RAG questions. Scalar quantization (IVF_SQ) runs only with
--scalar.

Community Edition caps the server at 5,000 vectors and counts deleted vectors until a restart, so
each variant starts from a restarted container with no other collections. The catalog collection
the main test uses is rebuilt at the end.

  uv run vector_side.py
  uv run vector_side.py --scalar
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from actian_vectorai import Distance, IndexType, IvfConfigDiff, PointStruct, VectorAIClient, VectorParams

from qpick import corpus, stack
from qpick.memory import GIB, container_cgroups

COL = "vq_test"


def restart_vectorai():
    subprocess.run(["docker", "restart", "vectorai"], check=True, capture_output=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            with VectorAIClient(stack.VECTORAI) as db:
                db.collections.list()
                return
        except Exception:
            time.sleep(2)
    raise SystemExit("VectorAI DB did not come back after a restart")


def variant_args(kind: str, n: int, dim: int) -> dict:
    if kind == "hnsw":
        return {}
    ivf = IvfConfigDiff.for_collection_size(n)
    if kind == "ivf_flat":
        return {"index_type": IndexType.INDEX_TYPE_IVF_FLAT, "ivf_config": ivf}
    if kind == "ivf_pq":   # 96 subvectors x 8 bits = 96 bytes per vector, against 3,072 bytes of float32
        return {"index_type": IndexType.INDEX_TYPE_IVF_PQ, "ivf_config": ivf,
                "extra_params_json": json.dumps({"pq_m": dim // 8, "pq_bits": 8})}
    if kind == "ivf_sq":
        return {"index_type": IndexType.INDEX_TYPE_IVF_SQ, "ivf_config": ivf}
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scalar", action="store_true", help="also test IVF with scalar quantization")
    args = ap.parse_args()

    cache = Path("results/cache")
    recs = corpus.records()
    vectors = np.load(cache / f"catalog-{stack.EMBED_MODEL}-{len(recs)}.npy")
    rag_qs = [q for q in corpus.rag_questions(recs) if q["rec"] is not None]
    qv = stack.embed([q["q"] for q in rag_qs], "search_query")
    rng = np.random.default_rng(0)
    probe = vectors[rng.choice(len(vectors), 200, replace=False)]
    exact = np.argsort(-(probe @ vectors.T), axis=1)[:, :10]

    kinds = ["hnsw", "ivf_flat", "ivf_pq"] + (["ivf_sq"] if args.scalar else [])
    out = []
    for kind in kinds:
        restart_vectorai()
        with VectorAIClient(stack.VECTORAI) as db:
            for name in db.collections.list():
                db.collections.delete(name)
        restart_vectorai()   # deleted vectors stop counting only after a restart
        cg = container_cgroups(["vectorai"]).get("vectorai")
        base = int(cg.read_text()) if cg else None
        with VectorAIClient(stack.VECTORAI) as db:
            db.collections.create(COL, vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.Cosine),
                                  **variant_args(kind, len(vectors), vectors.shape[1]))
            pts = [PointStruct(id=i, vector=v.tolist()) for i, v in enumerate(vectors)]
            for i in range(0, len(pts), 500):
                db.points.upsert(COL, pts[i:i + 500])
            if kind != "hnsw":
                db.vde.optimize(COL, wait=True)   # trains the IVF centroids and, for PQ/SQ, the codebooks
            db.vde.wait_for_ready(COL)
            time.sleep(5)
            stats = db.vde.get_stats(COL)
            hits = []
            for i in range(0, len(probe), 100):
                hits += db.points.search_batch(COL, [{"vector": v.tolist(), "limit": 10} for v in probe[i:i + 100]])
            rag = db.points.search_batch(COL, [{"vector": v.tolist(), "limit": stack.TOP_K} for v in qv])
        recall = float(np.mean([len({int(h.id) for h in hs} & set(e.tolist())) / 10 for hs, e in zip(hits, exact)]))
        rag_hit = sum(q["rec"] in {int(h.id) for h in hs} for q, hs in zip(rag_qs, rag))
        row = {"index": kind, "vectors": len(vectors), "index_memory_mib": round(stats.index_memory_bytes / 2**20, 2),
               "storage_mib": round(stats.storage_bytes / 2**20, 2),
               "container_growth_mib": round((int(cg.read_text()) - base) / 2**20, 1) if cg else None,
               "recall_at_10": round(recall, 4), "rag_top3_hits": f"{rag_hit}/{len(rag_qs)}"}
        out.append(row)
        print(row)

    Path("results/vector_side.json").write_text(json.dumps(out, indent=1))
    restart_vectorai()
    with VectorAIClient(stack.VECTORAI) as db:
        for name in db.collections.list():
            db.collections.delete(name)
    restart_vectorai()
    stack.build_index(recs, cache)   # put the catalog back for run.py
    print(f"Wrote results/vector_side.json; catalog rebuilt. Raw float32 vectors: "
          f"{vectors.nbytes / 2**20:.1f} MiB ({vectors.nbytes / GIB:.3f} GiB)")


if __name__ == "__main__":
    main()
