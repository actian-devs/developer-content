# EmbeddingGemma 2 multimodal search on VectorAI DB

Reproducible benchmarks for the article “How Does EmbeddingGemma 2 Perform on Multimodal Search?”. The included public sample is sourced from pinned Raspberry Pi documentation and includes text, official images, generated PDF pages, and short page-based video clips. The preparation script also generates synthesized voice after eSpeak NG is installed. It replaces personal sample data; do not commit personal documents, media, or embedding caches.

## Dataset and attribution

The corpus contains 14 distinct document assets from Raspberry Pi's documentation repository, pinned to commit [`7376533ab86ea07acc77ef6a9f3c3e57fcff833d`](https://github.com/raspberrypi/documentation/commit/7376533ab86ea07acc77ef6a9f3c3e57fcff833d), plus 10 selected official images. The upstream documentation is CC BY-SA 4.0. See [public_data/DATASET.md](public_data/DATASET.md) for per-asset source links and [public_data/UPSTREAM_LICENSE.md](public_data/UPSTREAM_LICENSE.md) for the upstream license.

`prepare_public_media.py` derives accessible PDFs and narration from the source docs, creates three-second videos showing the first rendered PDF page, and synthesizes query recordings from the hand-authored questions. These are benchmark derivatives, not original Raspberry Pi video or human-recorded speech. The voice clips are converted to mono 16 kHz WAV. The generated media inherits attribution obligations; retain the included dataset metadata when redistributing.

The prepared [`queries.json`](queries.json) contains 122 labels: 42 hand-authored text-to-manual-text labels and 80 generated multimodal labels. It covers text, photo, voice, and video queries against manual text, rendered pages, voice, and video targets. The current manifest has 136 records (49 manual-text page records, 49 rendered pages, 10 photos, 14 voice clips, and 14 videos). Ingest splits those 49 text records into 78 vectors, for 165 vectors total. Query labels point to the owning document asset. Review labels and note the generated-media limitations before publishing benchmark results.

## Setup and evaluation

See the complete, ordered workflow in [RUN-GUIDE.md](RUN-GUIDE.md). Quick start on Ubuntu:

```bash
uv sync
docker compose up -d
uv run hf download google/embeddinggemma-2
uv run smoke_test.py

# Public corpus and all supported media (Ubuntu packages)
sudo apt-get install -y espeak-ng ffmpeg fonts-dejavu-core
uv run fetch_public_dataset.py
uv run prepare_public_media.py
uv run make_manifest.py
uv run ingest.py --manifest manifest.json --recreate

# Evaluate hit-rate first. The filter cache is independent from this corpus.
uv run eval_hitrate.py --queries queries.json --k 5
uv run filter_check.py
uv run admin.py drop field_search
uv run filter_compare.py --sizes 1000 2500 5000 --filter-share 0.01 \
  --hardware "AMD Ryzen 5 2500U, 8 GB RAM, no CUDA GPU, float32"
# Restore field_search before inspecting hard cases.
uv run ingest.py --from-cache
uv run hard_case.py --queries queries.json --k 5
```

The commands above use the existing `filter_embeddings.npy` and `filter_meta.json`. If either file is missing or you need to rebuild the filter cache, run `uv run filter_corpus.py --source filter_text --target 5000 --chunk-words 10` on a directory containing enough PDF text that you have permission to process. Do not publish private source PDFs or their contents. The filter cache is independent from the multimodal ingest cache. Dropping `field_search` before comparison frees collection capacity on Community Edition; restore it from the validated ingest cache afterward. If the cache does not match the current manifest, rebuild with `uv run ingest.py --manifest manifest.json --recreate` instead of `--from-cache`.

## Scripts

- `fetch_public_dataset.py` fetches the pinned, attributed source documents and ten official images.
- `prepare_public_media.py` builds PDFs, page-based MP4s, narration, and query audio. Requires eSpeak NG, FFmpeg, and DejaVu Sans.
- `make_manifest.py` creates `manifest.json` and `pages.json`; the former is used by ingest and the latter by the optional ColPali comparison.
- `ingest.py` embeds manifest records and uploads them to `field_search`; `--from-cache` verifies that the local cache still agrees with the manifest.
- `eval_hitrate.py` writes hit-rate results to `hitrate.csv`.
- `hard_case.py` prints each query that misses a relevant asset in the top-k, plus ranked results and per-modality-pair failure totals.
- `filter_check.py` checks exact text and boolean payload filtering on temporary vectors.
- `filter_compare.py` compares FAISS HNSW post-filtering, FAISS HNSW with an ID selector, and VectorAI DB payload filtering. It reports recall, median latency, returned/off-filter counts, index memory, and VectorAI's indexed count.
- `colpali_compare.py` is optional and requires `uv sync --extra colpali`; it compares client-side ColPali page scoring with EmbeddingGemma 2 on manual pages.

## Important benchmark notes

- Record hardware and model/runtime versions with every result. Do not mix CPU and GPU results in one table.
- CPU inference uses float32; `embed.py` uses bfloat16 only when a CUDA GPU supports it and rejects non-finite embeddings.
- The model card lists a shared 8,192-token context, 280 image tokens by default, video sampling at 1 frame per second with about 58 frames, and audio at 25 tokens per second with mono 16 kHz input. Ingest rejects audio longer than 327 seconds.
- `filter_compare.py` prints `filter_compare v3`. It waits for the indexed vector count to match the requested corpus and reports off-filter results. Run `filter_check.py` before interpreting the filtering results.
- The database filter comparison is independent from relevance hit-rate and hard-case tests. It uses synthetic vectors and payloads, whereas the relevance evaluation uses the public corpus and query labels.
- Filter comparison datasets are not included. `filter_corpus.py` requires enough distinct text chunks for the requested vector count; a 5,000-vector run will not work with only the 14-document relevance corpus.

## Pinned components

| Component | Version / source |
|---|---|
| Model | `google/embeddinggemma-2`, Hugging Face snapshot `914f7f89142e33e77833254d9c9b90c3cef7303b` |
| sentence-transformers | 6.1.0 |
| actian-vectorai-client | 1.0.3 |
| faiss-cpu | 1.15.1 |
| Optional ColPali engine | 0.3.18 |
| ColPali checkpoint | `vidore/colpali-v1.3` |
| VectorAI DB image | `actian/vectorai:latest`, previously recorded image digest `sha256:b01cbb6b407129e722fe7017d6d93afa8648729b625946cf0e546858b8edc6c0` |

`docker-compose.yml` currently uses the mutable `actian/vectorai:latest` tag; the digest above is the image digest recorded for the earlier run, not a digest-enforced Compose setting. To record the image actually present on your machine, run `docker image inspect actian/vectorai:latest --format '{{json .RepoDigests}}'` and update this table if it differs.
