# VectorAI DB memory and disk benchmark

Measures how much memory and disk [Actian VectorAI DB](https://www.actian.com/vectorai/) uses as a collection grows, comparing **float32** storage with **int8** scalar quantization. It accompanies the article *How Much Memory Does Your Vector Database Need?*

The script reports **measured values only**. It does not apply any sizing formula or calculator, so you can compare its output with whatever estimate you like (see [Comparing with a calculator](#comparing-with-a-calculator)).

## What it measures

For each precision, the script:

1. Restarts the VectorAI DB container, so the baseline is a fresh idle process.
2. Records baseline memory and disk.
3. Creates one collection and records an **empty-collection** reading (the fixed cost).
4. Ingests random vectors up to each checkpoint (1,000 / 10,000 / 100,000 by default), flushes, waits, and records memory and disk **growth over the idle baseline**.
5. Deletes the collection.

It then fits a straight line, `growth(N) = fixed + slope × N`, and reports the per-vector slope next to the raw size of the vectors, `N × D × B` (vectors × dimensions × bytes per dimension).

| Setting | Value |
|---|---|
| Dimensions | 384 (all-MiniLM-L6-v2 size), configurable with `DIM` |
| Distance | Cosine |
| Index | Default HNSW settings |
| Vectors | Random Gaussian, seed 42 |
| Payload | `{"i": <id>}` on every point |
| float32 | Default storage |
| int8 | Scalar quantization, `quantile=0.99`, `always_ram=True` |

Memory is read from the container's cgroup (`anon` on cgroup v2, `rss` on v1), which excludes page cache. If the cgroup file isn't readable, it falls back to `docker stats`. The source used is printed and written to the report. Each reading is the median of 3 samples one second apart.

Disk is measured with `du -sk` on the container's mounted data directories.

## Requirements

- Python 3.10+
- Docker, with the VectorAI DB container running
- `pip install actian-vectorai numpy`

Developed against VectorAI DB 1.0.3 and the matching Python SDK. Other SDK versions may differ. For example, `VectorParams` rejected `on_disk` and `datatype` in this version, so the script doesn't pass them.

### Vector limits

Per Actian's Community Edition page, the 30-day trial allows up to 1 million vectors. The default checkpoints (up to 100,000) need the trial limit.

The script deletes each collection when it finishes, so it never holds more than the largest checkpoint.

## Usage

```bash
pip install actian-vectorai numpy

# find the container name and its data mounts
docker ps
docker inspect <container> --format '{{json .Mounts}}'

CONTAINER=<container> python bench_model.py

# macOS: keep the machine awake during long runs
caffeinate -i env CONTAINER=<container> python bench_model.py
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `CONTAINER` | `vectorai` | Docker container name |
| `SERVER` | `localhost:6574` | gRPC address of the database |
| `DATA_DIR` | auto-detect | Container-side data path for the disk measurement. Empty means every mount destination is measured. |
| `DIM` | `384` | Embedding dimensions |

Edit the constants at the top of `bench_model.py` to change `CHECKPOINTS`, `PRECISIONS` or `BATCH`.

## Output

| File | Contents |
|---|---|
| `benchmark_model.md` | Article table, fit table and per-checkpoint table (paste-ready Markdown) |
| `benchmark_model.html` | The same tables plus a chart of measured points and the fitted line |
| `benchmark_model_results.json` | Every raw reading, saved after each precision so a failure keeps earlier results |

The console prints progress, the ingest rate for each checkpoint, and the memory source.

## Example results (single run)

One run on an Apple M3 in Docker, VectorAI DB 1.0.3, 384 dimensions. Values are growth over an idle, freshly restarted container. Treat them as a guide, not a guarantee.

| Vectors | Precision | Raw (N·D·B) | Measured memory | Measured disk |
|---|---|---|---|---|
| 1,000 | Float32 | 1.46 MiB | 39.4 MiB | 4.4 MiB |
| 1,000 | Int8 | 0.37 MiB | 29.9 MiB | 4.9 MiB |
| 10,000 | Float32 | 14.65 MiB | 133.8 MiB | 60.0 MiB |
| 10,000 | Int8 | 3.66 MiB | 83.4 MiB | 24.8 MiB |
| 100,000 | Float32 | 146.48 MiB | 748.9 MiB | 488.5 MiB |
| 100,000 | Int8 | 36.62 MiB | 345.9 MiB | 435.8 MiB |

Observations from this run:

- An empty collection costs about 15 MiB of memory.
- Memory added per vector falls as the collection grows, so a single straight-line slope describes the data only roughly.
- Int8 reduced measured memory by about 1.3x, 1.6x and 2.2x at the three sizes, well short of the 4x reduction in raw vector bytes.
- Int8 saved little disk at 100,000 vectors, and the 10,000-vector disk readings look inconsistent with their neighbours. Repeat the runs before relying on the disk column.

## Comparing with a calculator

The script intentionally contains no estimate. To compare with [Actian's sizing calculator](https://actian-devs.github.io/vector-db-calculator/), set:

| Control | Setting |
|---|---|
| Vectors to store | 1K, 10K and 100K |
| Embedding dimensions | 384 |
| Growth in 12 months | flat |
| Assumptions → Memory headroom | 1.0 for a like-for-like comparison with measured use |

The calculator's own page describes its figures as rule-of-thumb estimates rather than benchmark results, and it reports raw int8 vector size only, with no int8 index estimate. The calculator uses decimal MB, while this script reports MiB (1 MiB = 1.048576 MB).

## Limitations

- Single run per cell on one machine. Memory readings varied by roughly 3 MiB between runs in earlier tests, so repeat each cell and report the median.
- Random Gaussian vectors, not real embeddings. Real data may index differently.
- The script waits 30 seconds after each checkpoint of 50,000 or more vectors. If indexing continues beyond that, the reading is low. A longer wait, or polling until memory stabilises, is a straightforward improvement.
- Memory and disk are measured for the whole container, so they include everything the server allocates, not only the index.
- Docker Desktop on macOS runs containers in a VM. Check that its memory limit is large enough for your largest checkpoint, or the container may be killed mid-run.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `du: cannot access '<path>'` | `DATA_DIR` is a host path. Use the container-side path from `docker inspect ... .Mounts` (the `Destination`). |
| `VectorParams ... Extra inputs are not permitted` | Your SDK version doesn't accept that argument. Check the installed `VectorParams` fields. |
| Run stops near 5,000 vectors | The server is on the Community Edition cap. Use the trial limit or lower `CHECKPOINTS`. |
| `server did not come back after restart` | The container is slow to start or crashed. Check `docker logs <container>`. |
| gRPC `fork_posix` log lines | Harmless. The script sets `GRPC_VERBOSITY=ERROR` to silence them. |
