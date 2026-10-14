"""Is Jev any good for reranking? A quick, reproducible benchmark.

First stage: dense search in Actian VectorAI DB + BM25, fused with reciprocal rank fusion (RRF).
Rerankers:   Jev (TypeSafe), a cross-encoder, and decider-4b served locally by Ollaya.
Reports:     nDCG@5/@10, Recall@10, first-stage recall, calibration (ECE + reliability curve),
             adaptive top-k, latency p50/p95, and Jev cost per 1,000 queries.

    uv run benchmark.py --queries 3     # smoke test, a few minutes
    uv run benchmark.py                 # article run: 50 queries per dataset, top 30 reranked

Scores are saved after every query, so an interrupted run resumes without re-paying for Jev.
"""
import argparse
import asyncio
import hashlib
import json
import math
import platform
import re
import time
import zipfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
from tqdm import tqdm  # noqa: E402

# ----------------------------------------------------------------------------- settings

SETTINGS = {
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "query_prefix": "Represent this sentence for searching relevant passages: ",
    "cross_encoder": "BAAI/bge-reranker-v2-m3",
    "jev_model": "jev-1.13.0",
    "local_model": "decider:4b",
    "ollaya_endpoint": "http://localhost:11435/v1/systemone",
    "vectorai_url": "localhost:6574",
    "first_stage_depth": 100,     # candidates retrieved by dense search and by BM25
    "rrf_k": 60,
    "calibration_bins": 10,
    "adaptive_chunk": 5,          # rerank in blocks of 5, in first-stage order...
    "adaptive_need": 5,           # ...until 5 candidates clear the confidence threshold
    "adaptive_thresholds": [0.5, 0.7, 0.9],
    "jev_usd_per_1m_input_tokens": 0.042,   # TypeSafe list price; output tokens are free
}

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
RERANKERS = ["jev", "cross-encoder", "decider-4b"]

# ----------------------------------------------------------------------------- data


def load_beir(name: str, data_dir: Path):
    """Download a BEIR dataset once; return corpus, queries, qrels and the zip's sha256."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{name}.zip"
    if not zip_path.exists():
        with requests.get(BEIR_URL.format(name), stream=True, timeout=120) as r:
            r.raise_for_status()
            zip_path.with_suffix(".part").write_bytes(r.content)
        zip_path.with_suffix(".part").rename(zip_path)
    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    folder = data_dir / name
    if not (folder / "corpus.jsonl").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)

    corpus = {}
    for line in open(folder / "corpus.jsonl", encoding="utf-8"):
        d = json.loads(line)
        corpus[str(d["_id"])] = f'{d.get("title") or ""}\n{d.get("text") or ""}'.strip()
    all_queries = {str(q["_id"]): q["text"] for q in map(json.loads, open(folder / "queries.jsonl", encoding="utf-8"))}
    qrels: dict[str, dict[str, int]] = {}
    with open(folder / "qrels" / "test.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, did, rel = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(rel)
    queries = {q: all_queries[q] for q in sorted(qrels) if q in all_queries}
    return corpus, queries, qrels, sha256


def corpus_embeddings(name: str, corpus: dict, embedder, data_dir: Path) -> np.ndarray:
    """Embed the full corpus once and cache it, rows aligned with list(corpus)."""
    tag = SETTINGS["embedding_model"].replace("/", "_")
    vec_path, ids_path = data_dir / f"{name}.{tag}.npy", data_dir / f"{name}.{tag}.ids.json"
    ids = list(corpus)
    if vec_path.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == ids:
        return np.load(vec_path)
    print(f"Embedding all {len(ids)} {name} documents once (cached for later runs)...")
    vecs = np.asarray(embedder.encode([corpus[d] for d in ids], normalize_embeddings=True,
                                      batch_size=32, show_progress_bar=True), dtype=np.float32)
    np.save(vec_path, vecs)
    ids_path.write_text(json.dumps(ids))
    return vecs


def subset_corpus(corpus: dict, queries: dict, qrels: dict, size: int, vecs: np.ndarray, embedder) -> dict:
    """Shrink the corpus without favouring either retriever.

    Keeps every judged document for the chosen queries, then adds hard distractors: each query's
    top dense and top BM25 matches over the full corpus, alternating and taken rank by rank across
    queries. Seeded random documents fill any remaining room. Same idea as the NanoBEIR subsets.
    """
    import random

    from rank_bm25 import BM25Okapi

    ids = list(corpus)
    keep = {d for q in queries for d in qrels[q] if d in corpus}
    if len(keep) > size:
        print(f"Note: {len(keep)} judged documents exceed --docs {size}; keeping all of them.")
    bm25 = BM25Okapi([tokenize(corpus[d]) for d in ids])
    qvecs = embedder.encode([SETTINGS["query_prefix"] + t for t in queries.values()], normalize_embeddings=True)
    ranked = []
    for text, qv in zip(queries.values(), qvecs):
        ranked.append(np.argsort(-(vecs @ np.asarray(qv, dtype=np.float32)))[:200])
        ranked.append(np.argsort(-bm25.get_scores(tokenize(text)))[:200])
    for rank in range(200):
        for order in ranked:
            if len(keep) >= size:
                break
            keep.add(ids[order[rank]])
    rest = sorted(set(ids) - keep)
    keep |= set(random.Random(42).sample(rest, max(0, min(size - len(keep), len(rest)))))
    return {d: corpus[d] for d in ids if d in keep}


def drop_stale_collections(client, dataset: str, current: str) -> None:
    """Delete this benchmark's older collections for the dataset, so they don't use up vector capacity."""
    try:
        names = [getattr(c, "name", c) for c in client.collections.list()]
    except Exception:
        names = [f"jev_bench_{dataset}"]
    for name in names:
        if isinstance(name, str) and name != current and (
            name == f"jev_bench_{dataset}" or name.startswith(f"jev_bench_{dataset}_")
        ):
            if client.collections.exists(name):
                print(f"Deleting old collection {name}")
                client.collections.delete(name)


# ----------------------------------------------------------------------------- first stage


def tokenize(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS   # installed with sentence-transformers

    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in ENGLISH_STOP_WORDS]


class FirstStage:
    """Dense search in VectorAI DB plus BM25, merged with the VectorAI DB SDK's RRF."""

    def __init__(self, dataset: str, corpus: dict, vecs: np.ndarray, embedder, client):
        from rank_bm25 import BM25Okapi

        self.doc_ids = list(corpus)
        # The name identifies the exact document set, so an index is only reused when it matches.
        digest = hashlib.sha1("\n".join(self.doc_ids).encode()).hexdigest()[:10]
        self.collection = f"jev_bench_{dataset}_{len(self.doc_ids)}_{digest}"
        self.vecs = vecs
        self.texts = [corpus[d] for d in self.doc_ids]
        self.embedder = embedder
        self.client = client
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

    def build_index(self) -> None:
        """Upload the document vectors, skipped when this exact collection is already complete."""
        from actian_vectorai import Distance, PointStruct, VectorParams

        if self.client.collections.exists(self.collection):
            if self.client.vde.get_vector_count(self.collection) == len(self.texts):
                return
            self.client.collections.delete(self.collection)
        self.client.collections.create(
            self.collection, vectors_config=VectorParams(size=int(self.vecs.shape[1]), distance=Distance.Cosine)
        )
        for start in tqdm(range(0, len(self.texts), 256), desc=f"index {self.collection}"):
            self.client.points.upsert(self.collection, [
                PointStruct(id=start + i, vector=v.tolist(), payload={"doc_id": self.doc_ids[start + i]})
                for i, v in enumerate(self.vecs[start : start + 256])
            ])
        self.client.vde.flush(self.collection)

    def retrieve(self, queries: dict[str, str]) -> dict[str, dict[str, list[str]]]:
        from actian_vectorai import reciprocal_rank_fusion
        from actian_vectorai.models.points import ScoredPoint

        depth = SETTINGS["first_stage_depth"]
        qvecs = self.embedder.encode(
            [SETTINGS["query_prefix"] + q for q in queries.values()], normalize_embeddings=True, batch_size=32
        )
        out = {"dense": {}, "hybrid": {}}
        for (qid, text), qvec in zip(queries.items(), qvecs):
            dense = self.client.points.search(self.collection, vector=qvec.tolist(), limit=depth) or []
            scores = self.bm25.get_scores(tokenize(text))
            top = np.argsort(-scores)[:depth]
            sparse = [ScoredPoint(id=int(i), version=0, score=float(scores[i]), payload={}) for i in top if scores[i] > 0]
            fused = reciprocal_rank_fusion([dense, sparse], limit=depth, ranking_constant_k=SETTINGS["rrf_k"])
            out["dense"][qid] = [self.doc_ids[int(p.id)] for p in dense]
            out["hybrid"][qid] = [self.doc_ids[int(p.id)] for p in fused]
        return out


# ----------------------------------------------------------------------------- rerankers


class SystemOneReranker:
    """Jev on TypeSafe, or a local model on Ollaya's TypeSafe-compatible endpoint.

    Pointwise mode scores each pair on its own, so probabilities are comparable across queries.
    """

    def __init__(self, name: str, model: str, endpoint: str | None = None, concurrency: int = 20):
        from jev_reranker import JevReranker

        self.name = name
        self.local = endpoint is not None
        kw = {"model": model, "mode": "pointwise", "max_concurrency": concurrency}
        if endpoint:   # local server: fail in minutes, not after 8 retries of 180 s
            kw.update(endpoint=endpoint, api_key="local", timeout=120, max_retries=1)
        self.client = JevReranker(**kw)   # hosted Jev reads TYPESAFE_API_KEY from .env

    async def score(self, query: str, docs: list[str], on_pair=None):
        """One request per document (the same calls pointwise mode makes), run concurrently."""
        async def one(doc):
            resp = await self.client.a_rerank(query, [doc], return_documents=False, detail=True)
            if on_pair:
                on_pair()
            usage = (resp.get("detail") or {}).get("usage") or {}
            return float(resp["results"][0]["score"]), int(usage.get("input_tokens") or 0)

        out = await asyncio.gather(*(one(d) for d in docs))
        return [s for s, _ in out], sum(t for _, t in out)


class CrossEncoderReranker:
    """Cross-encoder baseline; a sigmoid over its relevance logit gives a 0-1 score."""

    local = True

    def __init__(self, model: str, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name, self.torch, self.device = "cross-encoder", torch, device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()

    def _score(self, query: str, docs: list[str], on_pair=None) -> list[float]:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(docs), 16):
                batch = docs[i : i + 16]
                enc = self.tok([query] * len(batch), batch, padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(self.device)
                out += self.torch.sigmoid(self.model(**enc).logits.view(-1).float()).tolist()
                for _ in batch:
                    if on_pair:
                        on_pair()
        return out

    async def score(self, query: str, docs: list[str], on_pair=None):
        return await asyncio.to_thread(self._score, query, docs, on_pair), 0


def build_rerankers(names: list[str], device: str) -> dict:
    out = {}
    for name in names:
        if name == "jev":
            out[name] = SystemOneReranker("jev", SETTINGS["jev_model"])
        elif name == "decider-4b":
            out[name] = SystemOneReranker("decider-4b", SETTINGS["local_model"],
                                          endpoint=SETTINGS["ollaya_endpoint"], concurrency=4)
        elif name == "cross-encoder":
            out[name] = CrossEncoderReranker(SETTINGS["cross_encoder"], device)
    return out


async def rerank(reranker, queries, candidates, corpus, cache: Path) -> dict:
    """Score each query's candidates, appending one line per query to a cache file."""
    done = {}
    if cache.exists():
        done = {r["qid"]: r for r in map(json.loads, cache.read_text().splitlines()) if r}
    todo = [q for q in candidates if q not in done or set(done[q]["scores"]) != set(candidates[q])]
    if todo and reranker.local:   # one untimed pair, so model loading is not counted as latency
        print(f"Warming up {reranker.name} (the first call loads the model)...", flush=True)
        try:
            await reranker.score(queries[todo[0]], [corpus[candidates[todo[0]][0]]])
        except Exception as exc:
            raise SystemExit(f"{reranker.name} did not respond: {exc}\n"
                             "Check that `ollaya serve` is running and the model is pulled.") from exc
    bar = tqdm(total=sum(len(candidates[q]) for q in todo), desc=f"rerank {reranker.name}", unit="pair")
    with open(cache, "a", encoding="utf-8") as f:
        for q in todo:
            start = time.perf_counter()
            scores, tokens = await reranker.score(queries[q], [corpus[d] for d in candidates[q]],
                                                  on_pair=lambda: bar.update(1))
            rec = {"qid": q, "scores": dict(zip(candidates[q], scores)),
                   "latency_s": time.perf_counter() - start, "input_tokens": tokens}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[q] = rec
    bar.close()
    return {q: done[q] for q in candidates}


# ----------------------------------------------------------------------------- metrics


def ndcg(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    """trec_eval-style nDCG (linear gain), matching published BEIR numbers."""
    dcg = sum(qrel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((r for r in qrel.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    relevant = {d for d, r in qrel.items() if r > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def evaluate(rankings: dict[str, list[str]], qrels) -> dict:
    qs = list(rankings)
    mean = lambda f, k: float(np.mean([f(rankings[q], qrels[q], k) for q in qs]))  # noqa: E731
    return {"ndcg@5": mean(ndcg, 5), "ndcg@10": mean(ndcg, 10), "recall@10": mean(recall, 10)}


def apply_scores(order: list[str], scores: dict[str, float]) -> list[str]:
    """Reranked candidates by score (first-stage rank breaks ties), then the unscored tail."""
    scored = [d for d in order if d in scores]
    return sorted(scored, key=lambda d: -scores[d]) + [d for d in order if d not in scores]


def adaptive(order: list[str], scores: dict[str, float], depth: int, threshold: float) -> tuple[list[str], int]:
    """Rerank blocks in first-stage order until enough candidates clear the threshold."""
    chunk, need = SETTINGS["adaptive_chunk"], SETTINGS["adaptive_need"]
    used = 0
    while used < depth:
        used = min(used + chunk, depth)
        if sum(scores[d] >= threshold for d in order[:used]) >= need:
            break
    return apply_scores(order, {d: scores[d] for d in order[:used]}), used


def calibration(scores_by_q: dict, qrels) -> dict:
    """Reliability curve and expected calibration error. Unjudged pairs count as not relevant."""
    probs = np.array([p for q, s in scores_by_q.items() for p in s.values()])
    labels = np.array([float(qrels[q].get(d, 0) > 0) for q, s in scores_by_q.items() for d in s])
    bins = SETTINGS["calibration_bins"]
    idx = np.minimum((probs * bins).astype(int), bins - 1)
    curve, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = float(probs[m].mean()), float(labels[m].mean())
            ece += m.sum() / len(probs) * abs(acc - conf)
            curve.append({"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int(m.sum()),
                          "confidence": conf, "fraction_relevant": acc})
    return {"ece": float(ece), "pairs": int(len(probs)), "curve": curve}


def plot_reliability(dataset: str, calib: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
    for name, c in calib.items():
        ax.plot([p["confidence"] for p in c["curve"]], [p["fraction_relevant"] for p in c["curve"]],
                marker="o", label=f'{name} (ECE {c["ece"]:.3f})')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted relevance probability",
           ylabel="Fraction actually relevant", title=f"Reliability curve, {dataset}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- report


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def write_report(summary: dict, path: Path) -> None:
    s = summary["settings"]
    lines = ["# Benchmark results", "",
             f'{summary["run_at"]} on {summary["hardware"]}. {s["queries_per_dataset"]} test queries per dataset; '
             f'every reranker scores the top {s["rerank_depth"]} hybrid candidates.', ""]
    for ds, r in summary["datasets"].items():
        fs = r["first_stage_recall"]
        lines += [f"## {ds}", "",
                  f'{r["documents"]} documents indexed. Dataset sha256 `{r["sha256"]}`. First-stage recall, hybrid: '
                  f'R@{s["rerank_depth"]} {f4(fs["hybrid_at_depth"])}, R@100 {f4(fs["hybrid_at_100"])}. '
                  f'Dense only: R@100 {f4(fs["dense_at_100"])}.', "",
                  "| First stage and reranker | nDCG@5 | nDCG@10 | Recall@10 |", "|---|---|---|---|"]
        lines += [f'| {n} | {f4(m["ndcg@5"])} | {f4(m["ndcg@10"])} | {f4(m["recall@10"])} |' for n, m in r["lift"].items()]
        lines += ["", "| Reranker | ECE | Pairs | p50 ms | p95 ms | Cost per 1K queries (USD) |", "|---|---|---|---|---|---|"]
        for n, c in r["calibration"].items():
            p = r["performance"][n]
            cost = "n/a (own hardware)" if p["cost_per_1k"] is None else f'{p["cost_per_1k"]:.4f}'
            lines.append(f'| {n} | {f4(c["ece"])} | {c["pairs"]} | {p["p50_ms"]:.0f} | {p["p95_ms"]:.0f} | {cost} |')
        lines += ["", "| Reranker | Strategy | nDCG@10 | Mean candidates reranked | Cost per 1K queries (USD) |",
                  "|---|---|---|---|---|"]
        for n, rows in r["adaptive"].items():
            for a in rows:
                cost = "n/a" if a["cost_per_1k"] is None else f'{a["cost_per_1k"]:.4f}'
                lines.append(f'| {n} | {a["strategy"]} | {f4(a["ndcg@10"])} | {a["mean_reranked"]:.1f} | {cost} |')
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def versions() -> dict:
    out = {}
    for p in ["actian-vectorai-client", "jev-reranker", "sentence-transformers", "transformers", "torch", "rank-bm25"]:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


# ----------------------------------------------------------------------------- main


async def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["nfcorpus", "scifact"])
    ap.add_argument("--queries", type=int, default=50, help="test queries per dataset (0 = all)")
    ap.add_argument("--depth", type=int, default=30, help="hybrid candidates each reranker scores")
    ap.add_argument("--docs", type=int, default=2400,
                    help="documents indexed per dataset: every judged document for the chosen queries plus "
                         "hard BM25 distractors. The default keeps both datasets under VectorAI DB Community "
                         "Edition's 5,000-vector limit (0 = full corpus, needs a licence)")
    ap.add_argument("--rerankers", nargs="+", default=RERANKERS, choices=RERANKERS)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    if not 0 < args.depth <= SETTINGS["first_stage_depth"]:
        ap.error(f"--depth must be between 1 and {SETTINGS['first_stage_depth']}")

    import torch
    from actian_vectorai import VectorAIClient
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    hardware = (torch.cuda.get_device_name(0) if device == "cuda"
                else f"Apple {platform.machine()} (MPS)" if device == "mps" else platform.processor() or "CPU")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}. Loading models...")
    embedder = SentenceTransformer(SETTINGS["embedding_model"], device=device)
    rerankers = build_rerankers(args.rerankers, device)
    summary = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "hardware": hardware,
               "python": platform.python_version(), "packages": versions(),
               "settings": SETTINGS | {"queries_per_dataset": args.queries or "all", "rerank_depth": args.depth,
                                       "docs_per_dataset": args.docs or "all"},
               "datasets": {}}

    with VectorAIClient(SETTINGS["vectorai_url"]) as client:
        for ds in args.datasets:
            print(f"\n=== {ds} ===")
            corpus, queries, qrels, sha = load_beir(ds, Path("data"))
            if args.queries:
                queries = dict(list(queries.items())[: args.queries])
            full_vecs = corpus_embeddings(ds, corpus, embedder, Path("data"))
            row = {d: i for i, d in enumerate(corpus)}
            if args.docs and args.docs < len(corpus):
                corpus = subset_corpus(corpus, queries, qrels, args.docs, full_vecs, embedder)
            print(f"{len(corpus)} documents, {len(queries)} queries, sha256 {sha}")

            fs = FirstStage(ds, corpus, full_vecs[[row[d] for d in corpus]], embedder, client)
            drop_stale_collections(client, ds, fs.collection)
            try:
                fs.build_index()
            except Exception as exc:
                if "capacity" not in str(exc).lower() and "limit" not in str(exc).lower():
                    raise
                raise SystemExit(
                    f"VectorAI DB is over its vector limit: {exc}\n"
                    "Community Edition stores at most 5,000 vectors. Restart it (docker compose restart vectorai); "
                    "if the count does not drop, reset it (docker compose down && rm -rf local_data && "
                    "docker compose up -d). Use --docs to index fewer documents per dataset."
                ) from exc
            cands = fs.retrieve(queries)
            top = {q: c[: args.depth] for q, c in cands["hybrid"].items()}

            lift = {"dense only": evaluate(cands["dense"], qrels), "hybrid (RRF)": evaluate(cands["hybrid"], qrels)}
            mean_recall = lambda run, k: float(np.mean([recall(run[q], qrels[q], k) for q in run]))  # noqa: E731
            result = {"sha256": sha, "queries": len(queries), "documents": len(corpus), "lift": lift, "calibration": {}, "performance": {},
                      "adaptive": {},
                      "first_stage_recall": {"hybrid_at_depth": mean_recall(cands["hybrid"], args.depth),
                                             "hybrid_at_100": mean_recall(cands["hybrid"], 100),
                                             "dense_at_100": mean_recall(cands["dense"], 100)}}

            for name, rr in rerankers.items():
                recs = await rerank(rr, queries, top, corpus, out / f"scores_{ds}_{name}.jsonl")
                scores = {q: recs[q]["scores"] for q in top}
                lift[f"hybrid + {name}"] = evaluate({q: apply_scores(cands["hybrid"][q], scores[q]) for q in top}, qrels)
                result["calibration"][name] = calibration(scores, qrels)

                cost = None
                if name == "jev":
                    tokens = np.mean([recs[q]["input_tokens"] for q in top])
                    cost = float(tokens * SETTINGS["jev_usd_per_1m_input_tokens"] / 1e6 * 1000)
                lat = np.array([recs[q]["latency_s"] for q in top]) * 1000
                result["performance"][name] = {"p50_ms": float(np.percentile(lat, 50)),
                                               "p95_ms": float(np.percentile(lat, 95)), "cost_per_1k": cost}

                rows = [{"strategy": f"fixed top-{args.depth}", "ndcg@10": lift[f"hybrid + {name}"]["ndcg@10"],
                         "mean_reranked": float(args.depth), "cost_per_1k": cost}]
                for t in SETTINGS["adaptive_thresholds"]:
                    runs, used = {}, []
                    for q in top:
                        runs[q], n = adaptive(cands["hybrid"][q], scores[q], args.depth, t)
                        used.append(n)
                    rows.append({"strategy": f"adaptive, threshold {t}", "ndcg@10": evaluate(runs, qrels)["ndcg@10"],
                                 "mean_reranked": float(np.mean(used)),
                                 "cost_per_1k": None if cost is None else cost * np.mean(used) / args.depth})
                result["adaptive"][name] = rows

            if result["calibration"]:
                plot_reliability(ds, result["calibration"], out / f"reliability_{ds}.png")
            summary["datasets"][ds] = result

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, out / "summary.md")
    print(f"\nDone. Open {out / 'summary.md'}")


if __name__ == "__main__":
    asyncio.run(main())
