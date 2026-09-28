# Offline visual defect search: CLIP + Actian VectorAI DB

Search a manufacturing defect archive by photograph or by plain-English
description, filtered by part category and defect type, entirely offline.
CLIP turns images and text into comparable 512-dimensional vectors; Actian
VectorAI DB stores them with metadata and searches them.

Companion repo for the tutorial *Build an Offline Visual Defect Search System
With CLIP and VectorAI DB*.

---

## Dataset licence: research and evaluation only

This project uses **MVTec AD**, released under
[CC BY-NC-SA 4.0](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).

**MVTec state the dataset is for non-commercial research use and may not be
used for commercial purposes.** Do not ship MVTec images inside a commercial
product, and do not redistribute them.

For commercial manufacturing use, point `ingest.py` at your own inspection
archive. Nothing here depends on MVTec. `scan_corpus()` reads any directory
laid out as:

```
<root>/<part_category>/<split>/<defect_type>/<image>.png
```

If your archive is organised differently, `scan_corpus()` in `ingest.py` is
the only function you need to change. It must return a list of dicts with
`image_path`, `part_category`, `defect_type`, and `is_defect`.

---

## Quickstart

```bash
# 1. Project and dependencies
uv init defect-search && cd defect-search
uv add torch transformers pillow actian-vectorai-client matplotlib
# or: pip install -r requirements.txt

# 2. CLIP weights (~605MB, one time, needs network)
uv run huggingface-cli download openai/clip-vit-base-patch32

# 3. Database
docker compose up -d
uv run smoke_test.py                    # expect server details, not an error

# 4. Corpus: download MVTec AD, unpack to ./mvtec_ad

# 5. Index (45-90 min on CPU, minutes on GPU)
uv run ingest.py --data ./mvtec_ad

# 6. Search
uv run query.py --text "surface scratch on metal component" --part-category metal_nut
uv run query.py --image ./line_capture_0412.png --defect-only

# 7. Optional: benchmark and chart
uv run benchmark.py --out benchmark.csv
uv run make_chart.py --csv benchmark.csv --hardware "Ryzen 7 5800X, 32GB RAM, CPU-only"
```

For a fully air-gapped machine, copy `~/.cache/huggingface` across and set
`HF_HUB_OFFLINE=1`.

---

## Files

| File | Purpose |
|---|---|
| `clip_encoder.py` | CLIP encoding into one 512-d space. Handles transformers 4.x and 5.x |
| `ingest.py` | Walks the archive, batch-encodes, upserts vectors with metadata |
| `query.py` | `search_by_text()` and `search_by_image()`, plus a CLI |
| `benchmark.py` | Latency percentiles and Recall@5 |
| `filter_recall_test.py` | Filtered-HNSW recall against `efSearch` |
| `make_chart.py` | Renders the latency chart from `benchmark.csv` |
| `smoke_test.py` | Confirms the database is reachable |
| `docker-compose.yml` | VectorAI DB with a persistent volume |

---

## Gotchas worth knowing

**transformers 5.x breaks most CLIP tutorials.** `get_image_features` now
returns a `BaseModelOutputWithPooling` instead of a tensor, so the usual
one-liner raises `AttributeError: 'BaseModelOutputWithPooling' object has no
attribute 'shape'`. The `_features()` helper in `clip_encoder.py` handles both.

**The collection name must match.** `query.py` uses `COLLECTION =
"defect_search"`, which is `ingest.py`'s default. Change one and change both,
or searches fail with a collection-not-found error while the data sits in the
database.

**The volume mount is what makes the index persist.** Remove
`./vectorai_data:/var/lib/actian-vectorai` from `docker-compose.yml` and a
`docker compose down` costs you the whole encoding run.

**Recall@5 must exclude the query image.** Otherwise every query retrieves
itself at similarity 1.0 and recall reads 100% regardless of model quality.

**Never compare text scores to image scores.** Text-to-image similarity sits
around 0.2-0.3; image-to-image sits near 0.9. Rank within one mode only.