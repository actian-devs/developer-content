# EmbeddingGemma 2 multimodal search on VectorAI DB

Companion repo for the article "How Does EmbeddingGemma 2 Perform on Multimodal Search?". It embeds manuals, part photos, video and voice notes into one VectorAI DB collection, measures hit rate per modality, compares VectorAI DB with FAISS under a selective filter, and compares EmbeddingGemma 2 with ColPali on manual pages.

## Pinned versions

| Component | Version | Source |
|---|---|---|
| Model | `google/embeddinggemma-2` | [model card](https://huggingface.co/google/embeddinggemma-2). Pin the commit hash you downloaded: `.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b` |
| sentence-transformers | 6.1.0 | [developer guide](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/) requires v6.1.0 or later |
| actian-vectorai-client | 1.0.3 | PyPI |
| faiss-cpu | 1.15.1 | PyPI |
| colpali-engine (optional) | 0.3.18 | PyPI. Needs transformers 5.3 or later |
| ColPali checkpoint | `vidore/colpali-v1.3` | [model card](https://huggingface.co/vidore/colpali-v1.3) |
| Dataset | `TODO` | Name and version of the manual, photo, video and voice set |
| VectorAI DB image | `actian/vectorai:latest` | Actian publishes no tag list. Record the image digest you ran: `actian/vectorai@sha256:b01cbb6b407129e722fe7017d6d93afa8648729b625946cf0e546858b8edc6c0` |

## Quickstart

Full walkthrough with every file, command and expected output: [RUN-GUIDE.md](RUN-GUIDE.md).

```bash
uv sync                                   # add --extra colpali for step 11
docker compose up -d                       # VectorAI DB on localhost:6584
uv run hf download google/embeddinggemma-2
uv run smoke_test.py                      # database + model check
uv run make_manifest.py --data data       # writes manifest.json and pages.json
uv run ingest.py --manifest manifest.json
uv run eval_hitrate.py --queries queries.json --k 5
uv run hard_case.py --queries queries.json
uv run admin.py drop field_search         # free the 5,000-vector cap for the benchmark
uv run filter_corpus.py --source filter_text --target 5000 --chunk-words 10
uv run filter_check.py                    # verify text and boolean payload filters
uv run filter_compare.py --sizes 1000 2500 5000 --data-dir ./vectorai_data --hardware "<your CPU, RAM, GPU>"
uv run ingest.py --from-cache             # restore field_search without re-embedding
uv run colpali_compare.py                 # GPU required
```

The Community Edition is free for up to 5,000 vectors, so keep the collection and the largest `--sizes` value at or under that.

## Filter benchmark

Copy PDF manuals into `filter_text/`, then build a distinct text-chunk corpus for the filter benchmark:

```bash
mkdir -p filter_text
uv run filter_corpus.py --source filter_text --target 5000 --chunk-words 10
uv run filter_check.py
uv run filter_compare.py --sizes 1000 2500 5000 --filter-share 0.01 \
  --data-dir ./vectorai_data --hardware "<your CPU, RAM, GPU>"
uv run ingest.py --from-cache
```

`filter_corpus.py` writes `filter_embeddings.npy` and `filter_meta.json`, separate from the `embeddings.npy` and `meta.json` cache used by `ingest.py`. The PDFs currently in `filter_text/` need `--chunk-words 10` to produce 5,000 chunks; with the default 300 words, they produce only 361. Choose a chunk size appropriate for your manuals, ensuring the resulting chunks are still meaningful for your benchmark. The script requires at least the requested number of non-empty PDF text chunks and does not duplicate chunks.

`filter_check.py` tests exact text and boolean payload filters on 200 temporary vectors, waits for all vectors to be indexed, and reports any off-filter results. Run it before trusting the comparison. `filter_compare.py` prints `filter_compare v3`, compares HNSW with M=32 and efSearch=16 in FAISS and VectorAI, waits for VectorAI's indexed count to match the corpus size, and reports returned counts and off-filter results. FAISS's ID selector is approximate on HNSW; its recall is not expected to be 1.0. VectorAI index memory comes from the database's collection statistics. The comparison controls filter selectivity with `--filter-share` (1% by default).

The comparison cache and ingest cache are independent. `ingest.py --from-cache` now verifies that `embeddings.npy` and `meta.json` match the modalities and vector count implied by `manifest.json` before uploading. If the old `filter_corpus.py` overwrote those files, rebuild the collection with `uv run ingest.py --manifest manifest.json --recreate`; that re-embeds the manifest and refreshes the ingest cache.

## What each script reports

- `eval_hitrate.py`: hit rate at k per query-modality to target-modality pairing, written to `hitrate.csv`.
- `filter_compare.py`: filtered recall at k, median latency, returned count, off-filter count and index size for FAISS HNSW post-filtering, FAISS HNSW with an `IDSelector`, and VectorAI DB with a payload filter. VectorAI index memory comes from the database's collection statistics.
- `colpali_compare.py`: hit rate at k and bytes per page for EmbeddingGemma 2 and ColPali on manual pages.

## Diagrams

`diagrams/shared-space.png` (Figure 1) and `diagrams/technician-query.png` (Figure 2), with SVG sources.

## Hardware notes

- bf16 needs a GPU that supports it. On CPU the script falls back to float32. The model card warns that float16 can return NaN, and `embed.py` raises if any embedding is not finite.
- Record the hardware for every number you publish and never mix CPU and GPU runs in one table.

## Input limits

Taken from the model card: 8,192 tokens shared across all modalities, 280 tokens per image by default (about 29 images), 140 tokens per video frame at 1 frame per second by default (about 58 frames), 25 tokens per second of audio (about 327 seconds), mono at 16 kHz. `ingest.py` rejects audio over 327 seconds. Cut video longer than 58 seconds into clips first, for example `ffmpeg -i in.mp4 -c copy -f segment -segment_time 55 clip_%03d.mp4`.

## Data

The scripts read three files you supply. Their schemas are in the docstrings of `ingest.py` and `eval_hitrate.py`. `make_manifest.py` builds `manifest.json` and `pages.json` from a `data/<asset_id>/` folder layout. `queries.example.json` shows the query format. Label the query set by hand and have a second person check it. Do not label with the model under test.

## Not yet verified

- No script in this repo has been run against a live VectorAI DB server or a downloaded EmbeddingGemma 2 checkpoint. SDK imports, method names and filter construction were checked against `actian-vectorai-client` 1.0.3, and the FAISS calls were run on synthetic vectors.
- `colpali_compare.py` scores pages client-side. The SDK exposes `MultiVectorConfig` with a `MaxSim` comparator, but server-side MaxSim has not been tested.
