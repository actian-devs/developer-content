# Filtered Vector Search Recall Test

Companion repo for the tutorial "Filtered Vector Search Without Losing Recall". It measures how many of the best ten results a filtered search returns, for a client-side post-filter and for the filters in Actian VectorAI DB, at five filter selectivities. You can run it as is, then point the same harness at your own data.

Every number in this README comes from one fresh run on one machine (AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx, 7.2 GiB RAM, no CUDA GPU) with `actian-vectorai-client` 1.0.3 and VectorAI DB image digest `sha256:0abd6c0d6011a4379c6df69d15f06c440dc4cbb8652ced75d0608b80f83f6466`. Treat the numbers as a method to repeat, not as a benchmark. The repo measures recall and result counts only, and it reports no latency or speed figures.

## What the test answers

1. How much recall does a post-filter lose as the filter narrows?
2. Does a filter passed to `points.search` return a full list of ten results?
3. Does a wider `hnsw_ef` or an overfetch recover missing results?
4. Do `must`, `should`, `must_not`, and `min_should` clauses keep every returned product inside your rule?

Terms used below
- **Selectivity** is the share of the catalog a filter admits. A 5% filter admits 250 of 5,000 products.
- **Recall at 10** is the share of the exact filtered top ten that a search returns. NumPy computes the exact top ten by ranking only the products that pass the filter.
- **Results returned** is the average number of products a search returned for a limit of 10.
- **Off-filter** counts returned products that break the filter. It should read 0.

## Prerequisites

| You need | Notes |
|---|---|
| `git` | To clone the repo |
| [`uv`](https://docs.astral.sh/uv/) | Installs Python 3.12 and the dependencies |
| Docker with Compose | Runs the VectorAI DB server |
| About 2 GB of free disk space | The embedding model, the Python packages, and the server image |
| Internet access on the first run | `catalog.py` downloads `all-MiniLM-L6-v2` from Hugging Face |

A GPU is not required. Embedding 5,100 short texts runs on a CPU.

## Quick start

```bash
git clone TODO-repo-url filtered-vector-search-recall
cd filtered-vector-search-recall

uv sync --extra plot                  # creates .venv and installs the pinned dependencies
docker compose up -d                  # starts VectorAI DB on localhost:6574

uv run catalog.py                     # builds catalog.npz and catalog.json
uv run load.py                        # creates the "products" collection and loads the catalog
uv run measure.py --hardware "<CPU, RAM, GPU>" \
    --hnsw-ef 64 256 1024 --overfetch 5 20 --probe-limits 10 50 200 1000
uv run plot.py                        # writes images/recall-by-selectivity.png
uv run clauses.py                     # must, should, must-not, and min_should on a real query
```

`--hardware` is required so that every result file records where it came from. Record the image digest too.

```bash
docker image inspect actian/vectorai:latest --format "{{index .RepoDigests 0}}"
```

If you prefer to build the project file by file instead of cloning, follow [RUN-GUIDE.md](RUN-GUIDE.md). It embeds every file and ends at the same results.

## Repository layout

| Path | Purpose |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version` | Pinned dependencies and Python 3.12 |
| `docker-compose.yml` | VectorAI DB image, ports 6573, 6574, and 6575, and a local data volume |
| `catalog.py` | Generates the seeded catalog and queries, embeds them, and writes `catalog.npz` and `catalog.json` |
| `common.py` | Shared settings, catalog loading, `count_matching`, and version reporting |
| `load.py` | Creates the collection and upserts the catalog |
| `measure.py` | Runs every method at every selectivity and writes `recall.csv` and `probe.csv` |
| `clauses.py` | Runs the combined and `min_should` filters and re-checks every payload |
| `plot.py` | Draws recall by selectivity from `recall.csv` |
| `recall.csv`, `probe.csv` | Generated measurement output (ignored by Git) |
| `images/` | The tutorial figures |
| `RUN-GUIDE.md` | The same project, built from scratch |

## Script reference

### `catalog.py`

Generates 5,000 products and 100 queries with a fixed seed, embeds them with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, normalized), and writes `catalog.npz` and `catalog.json`.

| Flag | Default | Meaning |
|---|---|---|
| `--n` | 5000 | Number of products |
| `--queries` | 100 | Number of queries, assigned to categories in turn |
| `--seed` | 7 | Random seed |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `--out` | `catalog` | Output file stem |

Each product has a category (10 categories), a title and description, a price, a rating, a condition (`new`, `refurbished`, or `used`), an in-stock flag, and an `idx`. Price is drawn independently of category and text, so a price ceiling changes how many products a filter admits and nothing else.

### `load.py`

Deletes the collection if it exists, creates it with the catalog's vector size and cosine distance, upserts the products in batches of 256, flushes, and waits until the collection is ready. Run it again any time you change the catalog. It accepts `--url` (default `localhost:6574`) and `--collection` (default `products`).

The collection uses the server's default index settings. The script sets only the vector size and the distance.

### `measure.py`

For each selectivity, it sets a price ceiling at the matching quantile, counts the admitted products with NumPy and with the server, and runs every method on all 100 queries.

| Method | What it runs |
|---|---|
| `post_filter` | An unfiltered `points.search` with `limit=10`, then a Python check on each payload |
| `vectorai_filter` | `points.search` with a `FilterBuilder` filter and `limit=10` |
| `vectorai_ef<N>` | The same filtered search with `SearchParams(hnsw_ef=N)` |
| `vectorai_overfetch<F>` | The filtered search with `limit=10 * F`, keeping the first ten |

| Flag | Default | Meaning |
|---|---|---|
| `--hardware` | required | A label such as `"AMD Ryzen 5 2500U, 8 GB RAM, no CUDA GPU"` |
| `--k` | 10 | Results per query |
| `--shares` | 0.5 0.2 0.1 0.05 0.01 | Selectivities to test |
| `--hnsw-ef` | 256 | One method per value |
| `--overfetch` | 20 | One method per factor |
| `--probe-limits` | none | Limits for the candidate probe, which writes `probe.csv` |
| `--out` | `recall.csv` | Result file |
| `--collection`, `--url` | `products`, `localhost:6574` | Server settings |

Output columns in `recall.csv`

| Column | Meaning |
|---|---|
| `share`, `ceiling`, `admitted` | The target selectivity, the price ceiling, and the number of products the filter admits |
| `method` | One of the methods above |
| `recall_at_k` | Mean recall at k over 100 queries |
| `returned` | Mean number of results returned |
| `off_filter` | Returned products that break the filter, summed over all queries |

Output columns in `probe.csv`

| Column | Meaning |
|---|---|
| `share`, `limit` | The selectivity and the limit passed to the filtered search |
| `returned` | Mean filtered results returned over the first 20 queries |

If the server count differs from the NumPy count, the script prints `MISMATCH, check the collection`. Fix that before you read any recall number.

### `clauses.py`

Runs two filters on one query and prints how many products each admits, what each returned, and whether each returned payload passes the rule in Python (`ok` or `BAD`).

- Combined filter. The product must be electronics, it should be cheap or highly rated, and it must not be refurbished.
- `min_should` filter. The product must be electronics and meet at least two of three preferences.

It accepts `--query` (default `noise cancelling headphones for travel`), `--collection`, and `--url`.

### `plot.py`

Reads `recall.csv` and draws recall at 10 against selectivity for the post-filter, the VectorAI DB filter, and up to two variants that differ from the plain filter curve. Variants identical to the plain filter, such as the `hnsw_ef` runs in the author's results, are left out. Use `--csv` and `--out` to change the paths. It needs the `plot` extra (`uv sync --extra plot`).

## Reference results

These are values from the author's fresh run, for 5,000 products and 100 queries, with recall at 10 and the average number of results returned in parentheses.

| Selectivity | Admitted | Post-filter | VectorAI DB filter | `hnsw_ef=1024` | 5x overfetch | 20x overfetch |
|---|---|---|---|---|---|---|
| 50% | 2,501 | 0.506 (5.1) | 1.000 (10.0) | 1.000 (10.0) | 1.000 (10.0) | 1.000 (10.0) |
| 20% | 1,000 | 0.195 (1.9) | 0.983 (10.0) | 1.000 (10.0) | 0.983 (10.0) | 0.999 (10.0) |
| 10% | 502 | 0.092 (0.9) | 0.938 (10.0) | 1.000 (10.0) | 0.938 (10.0) | 1.000 (10.0) |
| 5% | 250 | 0.043 (0.4) | 0.815 (10.0) | 1.000 (10.0) | 0.982 (10.0) | 1.000 (10.0) |
| 1% | 50 | 0.003 (0.0) | 0.881 (10.0) | 0.999 (10.0) | 1.000 (10.0) | 1.000 (10.0) |

What the results show
- The post-filter's recall tracks the filter's share, because it can only keep what the unfiltered top ten already holds.
- The VectorAI DB filter returned exactly ten results for every one of the 100 queries at all five selectivities. A separate per-query check found zero products that violated the price filter.
- Returning ten results did not guarantee exact top-ten recall: the default filtered-search recall was 0.983 at 20%, 0.938 at 10%, 0.815 at 5%, and 0.881 at 1%.
- Raising `hnsw_ef` to 1,024 raised recall to at least 0.999 at every selectivity. Overfetching by 20 also reached at least 0.999.
- In the limit probe, the first 20 queries returned the requested 10, 50, or 200 results at every selectivity. For a limit of 1,000, results were capped by the number of products admitted by the filter.

## Use your own data

1. Replace `generate_products` and `generate_queries` in `catalog.py`, or write your own `catalog.npz` with `vectors` and `queries` arrays and a `catalog.json` with a `products` list (each with an `idx` and the payload fields you filter on) and a `model` entry.
2. Change the filter field in `measure.py` (`price`) to a numeric field in your payload, or build the filter you use in production.
3. Run `load.py`, then `measure.py`, and compare `admitted` with the server count before you read recall.

Check three signals on every run
- Off-filter reads 0.
- Results returned, set against the admitted count, shows whether the list came back full.
- Recall at your narrowest filter shows where the list starts to lose the right products.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `connection refused` on `localhost:6574` | Run `docker compose ps`. If the container is not running, run `docker compose up -d` and wait a few seconds. |
| Port 6573, 6574, or 6575 already in use | Stop the other service or change the host side of the port mapping in `docker-compose.yml`, and pass `--url` to the scripts. |
| `UnimplementedError` on count | `count_matching` already falls back to `points.scroll`, so you should see no error. |
| `MISMATCH, check the collection` | Run `uv run load.py` again. The collection may be incomplete or stale. |
| A search returns an empty list right after loading | Run `uv run load.py` again. It waits for the index before it exits. |
| The embedding model fails to download | Check your connection and the Hugging Face status, then run `uv run catalog.py` again. |
| `ModuleNotFoundError: matplotlib` | Run `uv sync --extra plot`. |
| `measure.py` exits with `--hardware` required | Pass a hardware label with `--hardware`. |
| You want a clean start | Run `docker compose down`, delete `vectorai_data/`, then repeat the quick start. |

## Pinned versions

| Component | Version |
|---|---|
| Python | 3.12 |
| `actian-vectorai-client` | 1.0.3 |
| `sentence-transformers` | 6.1.0 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| VectorAI DB image | `actian/vectorai:latest`, digest `sha256:0abd6c0d6011a4379c6df69d15f06c440dc4cbb8652ced75d0608b80f83f6466` |

Actian publishes no tag list for the image, so record the digest you ran with every number you publish.

## Limits of this test

- One catalog, one embedding model, and one machine. Recall depends on your data, your filters, and your index settings.
- Price is independent of text by design. A filter that correlates with the query changes the picture, and the Google Research tutorial on filtered vector search lists data correlation as one of the factors.
- The collection uses default index settings, and the test did not try other index configurations.
- The test reports no latency. Measure the cost of overfetching on your own hardware before you raise the factor.