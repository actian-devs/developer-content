# Run guide

This guide covers the public multimodal corpus workflow for EmbeddingGemma 2 and VectorAI DB. Run the stages in order: prepare and ingest the public corpus, run relevance hit-rate, run the VectorAI filter check/comparison, restore the search collection, and inspect hard-case misses.

## 1. Prerequisites

- Ubuntu or another environment that can run this project's Python dependencies.
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).
- Docker Engine and Docker Compose for a local VectorAI DB server.
- eSpeak NG, FFmpeg, and DejaVu Sans for generating the public voice/video/PDF derivatives. On Ubuntu:

  ```bash
  sudo apt-get update
  sudo apt-get install -y espeak-ng ffmpeg fonts-dejavu-core
  ```

- Enough disk space for the model checkpoint, generated PDFs/page renders/audio/video, and database.

Install the project dependencies and start the database:

```bash
uv sync
docker compose up -d
uv run hf download google/embeddinggemma-2
uv run smoke_test.py
```

`smoke_test.py` checks the database connection and model availability. It does not ingest the evaluation corpus.

## 2. Fetch the public corpus and prepare its media

The fetcher downloads 14 Raspberry Pi documentation sources and 10 selected images from the repository pinned at commit `7376533ab86ea07acc77ef6a9f3c3e57fcff833d`. It writes source URLs and license metadata alongside each asset.

```bash
uv run fetch_public_dataset.py
uv run prepare_public_media.py
uv run make_manifest.py
```

The media-preparation step with eSpeak NG creates:

- One adapted PDF per document, rendered into searchable page images for the image modality.
- One eSpeak NG narration WAV per document and a separate synthesized recording of a hand-authored question for each asset. The WAV files are mono, 16 kHz PCM.
- One three-second MP4 per document showing the first page of its generated PDF. These are still-page benchmark clips, not original video footage.
- An expanded `queries.json`. It retains the 42 hand-authored text-to-manual-text labels and adds 80 multimodal records: 14 voice-to-text, 14 voice-to-voice, 14 text-to-page, 14 text-to-video, 14 video-to-page, and 10 photo-to-page. Total: 122 query/target labels.

If eSpeak NG is unavailable, you can explicitly prepare the other modalities with:

```bash
uv run prepare_public_media.py --skip-voice
```

This mode prints that voice was skipped, creates PDFs and videos, and adds 52 records (total 94) for the available modalities. It does not create narration or voice-query files. Install eSpeak NG and rerun `uv run prepare_public_media.py` to generate the full 122-label set including voice.

The current prepared snapshot includes all modalities: 14 PDFs, 49 rendered pages, 10 official images, 14 narration WAVs, 14 synthesized question WAVs, 14 page-based videos, and 122 queries. The manifest contains 136 records: 49 manual-text page records, 49 rendered page images, 10 photos, 14 voice clips, and 14 videos. Ingest splits the manual text into 78 chunks, so the example run reports 165 vectors embedded.

The source documentation and selected images are identified in [public_data/DATASET.md](public_data/DATASET.md), with the upstream license in [public_data/UPSTREAM_LICENSE.md](public_data/UPSTREAM_LICENSE.md). The derivatives require the original attribution to be retained. Read the source and generated-media notes before redistributing or publishing results.

`make_manifest.py` reports the number of assets and records per modality. The current prepared snapshot produces 136 records across `manual_text`, `manual_page`, `photo`, `voice`, and `video`. Exact counts can change if the source revision or documents change. It writes:

- `manifest.json`: records consumed by `ingest.py`.
- `pages.json`: rendered manual-page image paths consumed by the optional ColPali comparison.

The fetch and media-preparation commands can be rerun. They restore the pinned originals and regenerate the generated media and evaluation queries. They do not use or overwrite `embeddings.npy` or `meta.json`.

## 3. Ingest and run hit-rate first

Ingest the manifest into the collection, replacing any previous `field_search` collection:

```bash
uv run ingest.py --manifest manifest.json --recreate
```

This embeds every manifest record, creates `embeddings.npy` and `meta.json` as a local cache, and uploads the points. The script prints the point count. Keep the two cache files with the matching manifest; `--from-cache` checks their modality/vector counts before upload.

Run the relevance hit-rate evaluation before the filter benchmark:

```bash
uv run eval_hitrate.py --queries queries.json --k 5
```

The script prints hit rate by query-to-target modality and writes `hitrate.csv`. The hit-rate metric is whether any relevant `asset_id` appears in the filtered top-k results. It is not a ranking metric, and the evaluation labels point to a document asset rather than an individual page.

Review the query labels and the generated-media limitations before using these values in an article. Query questions and generated media are not independent human assessments; they are a small demonstrative benchmark, not a statistically representative corpus.

## 4. Run the database filter check and comparison

The filter benchmark is separate from the public relevance corpus. It uses synthetic vectors plus payload fields to compare filtering behavior; it does not measure document-search quality.

First, prepare at least 5,000 distinct text chunks from PDFs or text that you are permitted to process. For example:

```bash
mkdir -p filter_text
# Copy suitable, licensed PDF or text files into ./filter_text/
uv run filter_corpus.py --source filter_text --target 5000 --chunk-words 10
```

This writes `filter_embeddings.npy` and `filter_meta.json`, independently of the ingest cache. If there are not enough usable chunks, the script stops and reports the available count; add more source text or select a smaller target.

The current workspace already has a 5,000-vector filter cache. If those files are present and were prepared from permitted source text, skip the `filter_corpus.py` step and run the filter check below. The `filter_text/` folder and filter cache are local-only; do not publish personal documents or cache data.

Run a small payload-filter sanity check first:

```bash
uv run filter_check.py
```

It tests text and boolean payload filters on a temporary collection, waits for indexing, and reports any results that do not satisfy the requested filter. Do not treat the comparison as valid if the check reports off-filter results or fails.

To leave room for the temporary filter collections on VectorAI DB Community Edition, drop the relevance collection, run the comparison, then restore it:

```bash
uv run admin.py drop field_search
uv run filter_compare.py --sizes 1000 2500 5000 --filter-share 0.01 \
  --hardware "AMD Ryzen 5 2500U, 8 GB RAM, no CUDA GPU, float32"
uv run ingest.py --from-cache
```

Use your own hardware description. The comparison prints `filter_compare v3`, the corpus sizes, and rows for FAISS HNSW post-filtering, FAISS HNSW with an ID selector, and VectorAI DB payload filtering. Each row reports recall@k, returned results, off-filter count, p50 latency, and index size; VectorAI rows also report indexed vectors. The script uses HNSW (M=32, efSearch=16), waits for VectorAI indexing, and gets VectorAI index memory from database statistics.

`filter_compare.py` creates and deletes its own temporary collections named `filter_compare_<size>`; it does not delete `field_search`. Dropping `field_search` in the sequence above is only to free collection capacity. `filter_check.py` likewise uses a temporary collection. The `--data-dir` option is retained for CLI compatibility; index sizes are read from VectorAI statistics.

If the run fails after `field_search` was dropped, restore it using the matching ingest cache:

```bash
uv run ingest.py --from-cache
```

If that command reports that the cache does not match `manifest.json`, rebuild it instead:

```bash
uv run ingest.py --manifest manifest.json --recreate
```

## 5. Inspect hard-case misses last

After restoring `field_search`, print every missed query and its ranked results:

```bash
uv run hard_case.py --queries queries.json --k 5
```

The output includes each failed query, its expected asset ID(s), returned asset/modality/path/text, and per-modality-pair hit and failure counts. It exits normally after reporting misses; misses are the intended output for diagnosis. To inspect one query modality:

```bash
uv run hard_case.py --queries queries.json --k 5 --query-modality photo
```

The supported query modalities are `text`, `photo`, `voice`, and `video`. If the collection is missing, the script prints the correct cache restore or rebuild command instead of making a search request.

## 6. Optional ColPali comparison

ColPali is not required for EmbeddingGemma hit-rate or hard-case evaluation. Install its optional project extra, then run it against the generated page set:

```bash
uv sync --extra colpali
uv run colpali_compare.py --pages pages.json --queries queries.json --k 5
```

This comparison needs `pages.json` entries and relevant page queries. ColPali uses a substantially larger model and may need more memory than the CPU-only EmbeddingGemma run.

## Output and result checks

- `manifest.json`, `pages.json`, `queries.json`: data and relevance labels for the current corpus.
- `embeddings.npy`, `meta.json`: ingest cache; do not confuse these with `filter_embeddings.npy` and `filter_meta.json`.
- `hitrate.csv`: modality-pair hit-rate output.
- `filter_compare.py` console output: synthetic payload-filter benchmark.
- `hard_case.py` console output: per-query retrieval failures and ranked candidates.

Before reporting results, record the exact model revision, VectorAI DB image digest, package versions, machine hardware, and command arguments. Do not invent expected recall, latency, or index-size values: these must come from the run being reported. The model card lists a shared 8,192-token context, 280 image tokens by default, video sampling at 1 frame per second with about 58 frames, and audio at 25 tokens per second with mono 16 kHz input. Ingest rejects audio longer than 327 seconds.
