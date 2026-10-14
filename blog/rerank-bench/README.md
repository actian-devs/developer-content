# Is Jev Any Good for Reranking?

This benchmark compares Jev (TypeSafe) with a cross-encoder and a local model, decider-4b served by Ollaya, for reranking on BEIR NFCorpus and SciFact. Actian VectorAI DB runs the first-stage retrieval.


## What the benchmark does

1. Takes the first 50 test queries from each dataset and indexes 2,400 documents per dataset.
2. For each query, retrieves the top 100 documents from dense search in VectorAI DB and the top 100 from BM25, then merges them with the VectorAI DB SDK's reciprocal rank fusion (RRF, k = 60).
3. Has each reranker score the top 30 merged candidates. Every query-document pair is scored on its own and gets a relevance probability between 0 and 1.
4. Reports nDCG@5, nDCG@10, Recall@10, first-stage recall, calibration (expected calibration error and reliability curves), adaptive top-k, latency, and Jev cost per 1,000 queries.

### Why 2,400 documents per dataset

VectorAI DB Community Edition stores at most 5,000 vectors, so the benchmark indexes 2,400 documents per dataset (4,800 in total). Each subset keeps every judged document for the evaluated queries, then adds the top dense and top BM25 matches from the full corpus as hard distractors, alternating between the two so neither retriever is favored. The subset is deterministic, so every run indexes the same documents.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) to run VectorAI DB.
- [uv](https://docs.astral.sh/uv/) to manage Python. It installs Python 3.12, the version pinned in `.python-version`.
- A TypeSafe API key for the hosted Jev model.
- [Ollaya](https://ollaya.dev), only for the local model comparison.

## Setup

1. Clone only this folder and install the pinned dependencies from `uv.lock`:

```bash
   git clone --depth 1 --filter=blob:none --sparse https://github.com/actian-devs/developer-content.git
   cd developer-content
   git sparse-checkout set blog/rerank-bench
   cd blog/rerank-bench
   uv sync
```

2. Add your TypeSafe key:

```bash
   cp .env.example .env
```

   Open `.env` and replace `your-typesafe-api-key` with your key. Git ignores `.env`.

3. Start VectorAI DB:

```bash
   docker compose up -d
```

   The service runs as `linux/amd64`, so on Apple Silicon Docker runs it through emulation.

## Reproduce the article

### Main results

Jev and the cross-encoder on 50 queries per dataset:

```bash
uv run benchmark.py --rerankers jev cross-encoder
```

The first run downloads the datasets and models, embeds both corpora, and indexes them in VectorAI DB. Later runs reuse the cached embeddings and the index. In the published run, Jev cost about $0.91 per 1,000 queries, so this run costs under $0.10.

### Local model comparison

decider-4b runs slowly on laptop hardware: about 23 seconds per query-document pair on the MacBook Pro M3 used for the article. The comparison therefore uses a smaller slice, with all three rerankers scoring the same queries and candidates.

Start Ollaya in a separate terminal and keep it running:

```bash
curl -fsSL https://ollaya.dev/install.sh | OLLAYA_INSTALL_DIR=$HOME/.local sh
export PATH="$HOME/.local/bin:$PATH"
ollaya pull decider:4b
ollaya serve
```

Then run:

```bash
uv run benchmark.py --queries 25 --depth 10 --out results-parity
```

### Resuming a run

The script saves every score as it goes. If a run stops, run the same command again. Finished queries are reused, so Jev is never billed twice for the same pair.

## Outputs

Each run writes to its output folder (`results/` by default):

| File | Contents |
|---|---|
| `summary.md` | Lift, calibration, latency, cost, and adaptive top-k tables |
| `summary.json` | Every number, plus hardware, package versions, and settings |
| `reliability_<dataset>.png` | Reliability curves |
| `scores_<dataset>_<reranker>.jsonl` | Raw per-pair scores, latency, and token counts |

## Published results

`published-results/` contains the run reported in the article. Compare your `results/summary.md` with `published-results/summary.md`.

| Setting | Value |
|---|---|
| Date | 2026-10-14 |
| Hardware | Apple MacBook Pro M3 (MPS) |
| Python | 3.12.11 |
| VectorAI DB | `actian/vectorai:TAG`, Community Edition |
| Jev model | `jev-1.13.0` |
| Embedding model | `BAAI/bge-base-en-v1.5` |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` |
| Packages | actian-vectorai-client 1.0.3, jev-reranker 0.1.2, sentence-transformers 6.1.0, transformers 5.19.0, torch 2.14.1, rank-bm25 0.2.2 |
| NFCorpus zip sha256 | `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b` |
| SciFact zip sha256 | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` |

Each run prints the dataset sha256, so you can confirm you evaluated the same data.

### What can differ in your run

- **Ranking metrics** should match closely. The datasets, query selection, corpus subset, and models are fixed.
- **Jev scores** can shift if TypeSafe updates the hosted model behind `jev-1.13.0`.
- **Latency** depends on your hardware and network. Jev latency includes the round trip to TypeSafe's API from your location.
- **Cross-encoder latency** reflects the laptop GPU. A server GPU runs it much faster.

## Options

```bash
uv run benchmark.py --queries 100 --depth 50       # larger run
uv run benchmark.py --datasets scifact             # one dataset
uv run benchmark.py --rerankers jev                # one reranker
uv run benchmark.py --docs 0                       # full corpus, needs a VectorAI DB licence
```

