# Run Guide: EmbeddingGemma 2 Multimodal Search on VectorAI DB

This guide takes you from an empty folder to every number the article needs. Each step lists the files involved, the full code in those files, the command to run, and what you should see. The values in the expected output (scores, rates, timings, counts) depend on your data and hardware, so they appear as `x.xxx` or `N`. The format of each line is exactly what the script prints.

Work through the steps in order. Steps 1 to 9 are the core build. Step 10 (FAISS) and step 11 (ColPali) produce the comparison tables, and step 11 needs a GPU.

---

## Step 0: What You Need Before You Start

| Requirement | Why | Check it with |
|---|---|---|
| Linux, macOS, or Windows with WSL2 | All commands below are bash | `uname -a` |
| 16 GB RAM or more | EmbeddingGemma 2 has 740M parameters, about 3 GB of weights in float32 and 1.5 GB in bfloat16, plus the database and your media | `free -h` (Linux) |
| Docker with Compose v2 | Runs VectorAI DB | `docker compose version` |
| uv | Creates the Python environment and runs every script | `uv --version` |
| ffmpeg | Video decoding, and converting voice notes and clips to the formats the model accepts | `ffmpeg -version` |
| A Hugging Face account | Downloads the model weights | Sign in at huggingface.co |
| A GPU with bf16 support (optional for steps 1 to 10, required for step 11) | The model card warns against float16, so the scripts use bf16 on a GPU and float32 on CPU | Step 5 prints it |

Install uv and ffmpeg if you are missing them:

```bash
# uv (Linux and macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# ffmpeg
sudo apt install ffmpeg        # Ubuntu / Debian / WSL2
brew install ffmpeg            # macOS
```

---

## Step 1: Create the Project Folder

Every file in this guide lives in one folder. By the end of the guide it looks like this:

```text
embeddinggemma2-multimodal-search/
├── pyproject.toml          # dependencies (step 2)
├── docker-compose.yml      # VectorAI DB (step 3)
├── embed.py                # shared model wrapper, imported by every script (step 4)
├── smoke_test.py           # checks database + model (step 5)
├── make_manifest.py        # builds manifest.json and pages.json from data/ (step 6)
├── ingest.py               # embeds and upserts everything (step 7)
├── admin.py                # count or drop a collection (step 7)
├── queries.example.json    # format of the labelled query set (step 8)
├── eval_hitrate.py         # hit rate per modality (step 9)
├── hard_case.py            # photo -> manual page failure analysis (step 9)
├── filter_corpus.py        # builds 5,000 vectors for the filter test (step 10)
├── filter_check.py         # confirms the payload filter is applied (step 10)
├── filter_compare.py       # FAISS against VectorAI DB under a filter (step 10)
├── colpali_compare.py      # ColPali against EmbeddingGemma 2 on pages (step 11)
├── data/                   # your corpus (step 6)
├── queries/                # photos and voice clips used as queries (step 8)
└── vectorai_data/          # created by Docker, holds the database files
```

If you cloned the repo, all the files are already there and you can skip to step 2. If you are building it by hand, create the folder and add each file as the steps introduce it:

```bash
mkdir embeddinggemma2-multimodal-search && cd embeddinggemma2-multimodal-search
```

---

## Step 2: Install the Python Dependencies

`pyproject.toml` pins the versions the article reports. sentence-transformers 6.1.0 is the minimum version Google's developer guide gives for EmbeddingGemma 2, and the `[image,audio,video]` extras pull in the media loaders. `pymupdf` renders PDF manual pages to images in step 6. ColPali sits in an optional extra because it brings a 3B-parameter model you only need in step 11.

**File: `pyproject.toml`**

```toml
[project]
name = "embeddinggemma2-multimodal-search"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "sentence-transformers[image,audio,video]==6.1.0",
    "actian-vectorai-client==1.0.3",
    "faiss-cpu==1.15.1",
    "numpy",
    "soundfile",
    "pymupdf",
]

[project.optional-dependencies]
colpali = ["colpali-engine==0.3.18"]
```

Pin Python 3.12 and install:

```bash
uv python pin 3.12
uv sync
```

**What to expect:** uv creates `.venv/`, resolves the packages, and writes `uv.lock`. The first run downloads PyTorch, which is the largest package and takes a few minutes. The command ends with a line listing the installed packages, with no errors.

If you are on Linux with an NVIDIA GPU, confirm that PyTorch sees it and that it supports bf16:

```bash
uv run python -c "import torch; print('cuda', torch.cuda.is_available()); print('bf16', torch.cuda.is_available() and torch.cuda.is_bf16_supported())"
```

`cuda True` and `bf16 True` mean the scripts run in bfloat16 on the GPU. Anything else means they run in float32 on the CPU, which works for steps 1 to 10 but is slower.

---

## Step 3: Start VectorAI DB

The Compose file runs one container, exposes the three ports from the VectorAI DB Docker setup guide, accepts the End User License Agreement (EULA) through an environment variable, and mounts `./vectorai_data` so the collection survives restarts.

**File: `docker-compose.yml`**

```yaml
services:
  vectorai:
    image: actian/vectorai:latest
    container_name: vectorai
    ports:
      - "6573:6573"   # REST API
      - "6574:6574"   # gRPC API, used by the Python SDK
      - "6575:6575"   # local web UI
    volumes:
      - ./vectorai_data:/var/lib/actian-vectorai
    environment:
      - ACTIAN_VECTORAI_ACCEPT_EULA=YES
    restart: unless-stopped
```

Start it:

```bash
docker compose up -d
docker compose ps
```

**What to expect:** `docker compose up -d` pulls `actian/vectorai:latest` on the first run, then prints `Container vectorai  Started`. `docker compose ps` lists the `vectorai` container with status `Up` and the ports 6573 to 6575. A `vectorai_data/` folder appears next to the Compose file.

Actian publishes no list of version tags, so record the exact image you ran. The article's "Run It Yourself" section needs this digest:

```bash
docker image inspect actian/vectorai:latest --format '{{index .RepoDigests 0}}'
```

That prints `actian/vectorai@sha256:...`. Paste it into the README's version table.

The Community Edition is free for up to 5,000 vectors. Steps 7 and 10 both write to the database, so step 10 explains how to stay under that cap.

---

## Step 4: Download the Model and Add the Shared Wrapper

Download EmbeddingGemma 2 once, while the machine has network access. Every later script loads it from the local cache.

```bash
uv run hf auth login                       # paste a read token from huggingface.co/settings/tokens
uv run hf download google/embeddinggemma-2
```

**What to expect:** `hf download` prints a progress bar per file, then the local path, which ends in `.../models--google--embeddinggemma-2/snapshots/<commit-hash>`. That commit hash is the model version the article should pin, so copy it into the README. If the download returns a 401 or 403 error, open the model page on Hugging Face while signed in, accept any terms it shows, and run the command again.

`embed.py` is the only place that loads the model. Every other script imports from it, so all of them use the same dtype, the same prompts, and the same NaN check.

**File: `embed.py`**

```python
"""Shared EmbeddingGemma 2 wrapper. Every script imports from here."""
from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
DIM = 768  # native size; the model card also lists 512, 256 and 128


def load_model(text_only: bool = False) -> SentenceTransformer:
    # The model card warns that float16 can return NaN, so use bf16 where the
    # GPU supports it and float32 everywhere else.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    # text_only loads the 270M text model without the vision and audio encoders
    # (developer guide). Its vectors live in the same space as the full model's.
    config_kwargs = {"vision_config": None, "audio_config": None} if text_only else None
    return SentenceTransformer(MODEL_ID, model_kwargs={"torch_dtype": dtype}, config_kwargs=config_kwargs)


def _check(vec: np.ndarray) -> np.ndarray:
    if not np.isfinite(vec).all():
        raise ValueError("Embedding contains NaN or inf. Check the dtype (use bf16 or fp32).")
    return vec


def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    return _check(model.encode(text, prompt_name="SearchQuery", normalize_embeddings=True))


def embed_document(model: SentenceTransformer, text: str) -> np.ndarray:
    return _check(model.encode(text, prompt_name="Document", normalize_embeddings=True))


def embed_media(model: SentenceTransformer, kind: str, path: str) -> np.ndarray:
    """kind is 'image', 'audio' or 'video'. Media takes no prompt_name."""
    if kind not in {"image", "audio", "video"}:
        raise ValueError(f"unknown media kind: {kind}")
    return _check(model.encode({kind: str(path)}, normalize_embeddings=True))
```

Three details in this file decide whether the numbers are valid. `load_model` picks bfloat16 or float32 and never float16, because the model card warns that float16 can return NaN or silently degraded embeddings. Text queries use the `SearchQuery` prompt and text documents use `Document`, which are the retrieval prompts in Google's developer guide, while images, audio and video go in as a dictionary with no prompt. `_check` stops the run at the first non-finite vector, because a NaN vector raises no error by itself and quietly ruins every search that touches it.

---

## Step 5: Run the Smoke Test

The smoke test checks the database connection and the model in one run, before you spend time embedding a corpus. It embeds one query, one document and one image, and prints the shape and length of each vector.

**File: `smoke_test.py`**

```python
"""Check the database and the model before indexing anything."""
from __future__ import annotations

import sys

import numpy as np
from actian_vectorai import VectorAIClient
from PIL import Image

from embed import embed_document, embed_media, embed_query, load_model

with VectorAIClient("localhost:6574") as client:
    print("VectorAI DB:", client.health_check())

model = load_model()
print("model dtype:", next(model.parameters()).dtype)

q = embed_query(model, "replace the impeller seal")
d = embed_document(model, "To replace the impeller seal, drain the pump housing first.")
img_path = sys.argv[1] if len(sys.argv) > 1 else "smoke.png"
if len(sys.argv) == 1:
    Image.new("RGB", (224, 224), "white").save(img_path)
i = embed_media(model, "image", img_path)

for name, v in [("query", q), ("document", d), ("image", i)]:
    print(f"{name:<9} shape {v.shape}  norm {np.linalg.norm(v):.3f}  finite {np.isfinite(v).all()}")
print(f"query-document cosine {float(q @ d):.3f}")
```

Run it with a blank test image, or pass a real photo as the argument:

```bash
uv run smoke_test.py
uv run smoke_test.py data/pump-a12/photos/01.jpg     # after step 6
```

**What to expect:**

```text
VectorAI DB: {...server details...}
model dtype: torch.bfloat16          # or torch.float32 on CPU
query     shape (768,)  norm 1.000  finite True
document  shape (768,)  norm 1.000  finite True
image     shape (768,)  norm 1.000  finite True
query-document cosine x.xxx
```

Every vector must be shape `(768,)`, norm `1.000`, and `finite True`. The query and document are about the same task, so their cosine should be clearly positive. If the first line is a connection error, go back to step 3. If `finite` prints `False`, the dtype is wrong, so check the second line.

---

## Step 6: Build the Corpus and the Manifest

The corpus is one folder per asset under `data/`. The folder name becomes the `asset_id` that every query is labelled against, so name each one after the physical asset, for example `pump-a12` or `valve-b07`.

```text
data/
  pump-a12/
    manual/   service_manual.pdf      -> manual text, one record per page, plus each page as an image
    photos/   01.jpg 02.jpg           -> part photos
    video/    seal_replacement.mp4    -> repair clips, under 58 seconds each
    voice/    note_2026-09-14.wav     -> voice notes, 16 kHz mono, under 327 seconds each
  valve-b07/
    ...
```

Every subfolder is optional, so an asset with only a manual and photos is fine. Use at least 10 assets, because with fewer the filter in step 10 admits so much of the corpus that the test shows nothing. Use manuals, photos and recordings you have the rights to, and note the source and version, since the article pins the dataset.

The model has input limits that the media has to meet first. Audio must be mono at 16 kHz and at most 327 seconds, and video is sampled at one frame per second with a budget of about 58 frames. ffmpeg handles both:

```bash
# voice note from a phone (m4a, mp3, ...) to 16 kHz mono WAV
ffmpeg -i note.m4a -ac 1 -ar 16000 data/pump-a12/voice/note.wav

# long repair video into 55-second clips
ffmpeg -i repair_full.mp4 -c copy -f segment -segment_time 55 -reset_timestamps 1 \
       data/pump-a12/video/repair_%03d.mp4
```

`make_manifest.py` walks `data/`, renders every PDF page to a PNG in the asset's `pages/` folder, extracts each page's text, and writes two files. `manifest.json` lists every item to embed, and `pages.json` lists the page images that the ColPali comparison in step 11 reads.

**File: `make_manifest.py`**

```python
"""Build manifest.json and pages.json from a folder layout.

data/
  pump-a12/
    manual/*.pdf          -> manual_text (one record per page) + manual_page (page rendered to PNG)
    manual.txt            -> manual_text (optional, if you have text but no PDF)
    pages/*.png|jpg       -> manual_page (optional, pre-rendered page images)
    photos/*.png|jpg      -> photo
    video/*.mp4           -> video   (keep clips under ~58 s)
    voice/*.wav|flac      -> voice   (16 kHz mono, under 327 s)
  valve-b07/
    ...
The folder name under data/ becomes the asset_id.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf

IMAGE = {".png", ".jpg", ".jpeg"}
FOLDERS = {"pages": ("manual_page", IMAGE), "photos": ("photo", IMAGE),
           "video": ("video", {".mp4"}), "voice": ("voice", {".wav", ".flac"})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--dpi", type=int, default=144, help="resolution for rendered PDF pages")
    args = ap.parse_args()

    manifest, pages = [], []
    for asset_dir in sorted(p for p in Path(args.data).iterdir() if p.is_dir()):
        asset_id = asset_dir.name

        # PDFs: text per page, plus each page rendered to pages/<pdf>_pNNN.png
        for pdf in sorted((asset_dir / "manual").glob("*.pdf")):
            (asset_dir / "pages").mkdir(exist_ok=True)
            with pymupdf.open(pdf) as doc:
                for n, page in enumerate(doc, start=1):
                    text = page.get_text().strip()
                    if text:
                        manifest.append({"asset_id": asset_id, "modality": "manual_text",
                                         "text": text, "source": f"{pdf.name}#p{n}"})
                    page.get_pixmap(dpi=args.dpi).save(asset_dir / "pages" / f"{pdf.stem}_p{n:03d}.png")

        manual = asset_dir / "manual.txt"
        if manual.exists():
            manifest.append({"asset_id": asset_id, "modality": "manual_text",
                             "text": manual.read_text(), "source": "manual.txt"})

        for folder, (modality, suffixes) in FOLDERS.items():
            for f in sorted((asset_dir / folder).glob("*")):
                if f.suffix.lower() in suffixes:
                    manifest.append({"asset_id": asset_id, "modality": modality, "path": str(f)})
                    if modality == "manual_page":
                        pages.append({"asset_id": asset_id, "path": str(f)})

    Path("manifest.json").write_text(json.dumps(manifest, indent=2))
    Path("pages.json").write_text(json.dumps(pages, indent=2))
    counts: dict[str, int] = {}
    for rec in manifest:
        counts[rec["modality"]] = counts.get(rec["modality"], 0) + 1
    print(f"assets: {len({r['asset_id'] for r in manifest})}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<12} {v}")
    print(f"wrote manifest.json ({len(manifest)} records) and pages.json ({len(pages)} pages)")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run make_manifest.py --data data
```

**What to expect:**

```text
assets: N
  manual_page  N
  manual_text  N
  photo        N
  video        N
  voice        N
wrote manifest.json (N records) and pages.json (N pages)
```

Each asset now has a `pages/` folder of PNGs, one per PDF page. Open two or three of them to confirm they are readable. A modality missing from the list means its folder name or file extension did not match the layout above.

---

## Step 7: Embed and Index Everything

`ingest.py` reads `manifest.json`, embeds every record through `embed.py`, and upserts the vectors into one collection called `field_search`, with `asset_id`, `modality`, and either the file path or the text chunk in each point's payload. It also saves the vectors to `embeddings.npy` and the payloads to `meta.json`. Step 10 reads those two files so that FAISS and VectorAI DB index exactly the same vectors, and `--from-cache` reloads them without embedding again.

**File: `ingest.py`**

```python
"""Embed every asset in a manifest and upsert it into one VectorAI DB collection.

manifest.json is a list of records:
  {"asset_id": "pump-a12", "modality": "manual_text", "text": "..."}
  {"asset_id": "pump-a12", "modality": "manual_page", "path": "manuals/pump-a12/p03.png"}
  {"asset_id": "pump-a12", "modality": "photo",       "path": "photos/pump-a12/01.jpg"}
  {"asset_id": "pump-a12", "modality": "video",       "path": "video/pump-a12/repair.mp4"}
  {"asset_id": "pump-a12", "modality": "voice",       "path": "voice/pump-a12/note.wav"}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from actian_vectorai import (
    CollectionExistsError, Distance, PointStruct, VectorAIClient, VectorParams,
)

from embed import DIM, embed_document, embed_media, load_model

MEDIA_KIND = {"manual_page": "image", "photo": "image", "video": "video", "voice": "audio"}
MAX_AUDIO_SECONDS = 327   # 8K shared context at 25 audio tokens per second (model card)
MAX_WORDS_PER_CHUNK = 300  # keeps a text chunk far below the 8,192-token limit


def chunk_words(text: str, size: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)] or [""]


def check_audio_length(path: str) -> None:
    seconds = sf.info(path).duration
    if seconds > MAX_AUDIO_SECONDS:
        raise SystemExit(f"{path} runs {seconds:.0f}s, over the {MAX_AUDIO_SECONDS}s limit. Split it first.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--collection", default="field_search")
    ap.add_argument("--url", default="localhost:6574")
    ap.add_argument("--recreate", action="store_true", help="drop the collection first")
    ap.add_argument("--from-cache", action="store_true",
                    help="skip embedding and reload embeddings.npy + meta.json from an earlier run")
    args = ap.parse_args()

    if args.from_cache:
        vectors = list(np.load("embeddings.npy"))
        rows = json.loads(Path("meta.json").read_text())
        print(f"loaded {len(vectors)} cached vectors")
    else:
        rows, vectors = embed_manifest(args.manifest)
        np.save("embeddings.npy", np.stack(vectors))
        Path("meta.json").write_text(json.dumps(rows))

    upload(args, rows, vectors)


def embed_manifest(manifest: str) -> tuple[list[dict], list[np.ndarray]]:
    records = json.loads(Path(manifest).read_text())
    model = load_model()
    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    for n_rec, rec in enumerate(records, start=1):
        print(f"\rembedding {n_rec}/{len(records)} ({rec['modality']})", end="", flush=True)
        if rec["modality"] == "manual_text":
            for n, chunk in enumerate(chunk_words(rec["text"])):
                rows.append({"asset_id": rec["asset_id"], "modality": "manual_text", "chunk": n,
                             "text": chunk, "source": rec.get("source", "")})
                vectors.append(embed_document(model, chunk))
            continue
        kind = MEDIA_KIND[rec["modality"]]
        if kind == "audio":
            check_audio_length(rec["path"])
        rows.append({"asset_id": rec["asset_id"], "modality": rec["modality"], "path": rec["path"]})
        vectors.append(embed_media(model, kind, rec["path"]))

    print(f"\n{len(vectors)} vectors embedded")
    return rows, vectors


def upload(args: argparse.Namespace, rows: list[dict], vectors: list[np.ndarray]) -> None:
    with VectorAIClient(args.url) as client:
        if args.recreate:
            try:
                client.collections.delete(args.collection)
                print(f"Dropped '{args.collection}'")
            except Exception:
                pass
        try:
            client.collections.create(
                args.collection, vectors_config=VectorParams(size=DIM, distance=Distance.Cosine)
            )
        except CollectionExistsError:
            print(f"Collection '{args.collection}' exists, appending")
        points = [PointStruct(id=i, vector=v.tolist(), payload=row)
                  for i, (row, v) in enumerate(zip(rows, vectors))]
        for start in range(0, len(points), 256):   # batches keep each gRPC message small
            client.points.upsert(args.collection, points=points[start : start + 256])
        print(f"{client.points.count(args.collection)} points in '{args.collection}'")


if __name__ == "__main__":
    main()
```

The chunk size of 300 words keeps every text chunk far below the shared 8,192-token context. The audio check rejects voice notes over 327 seconds instead of letting the model truncate them silently. The upserts run in batches of 256 points, because one message holding thousands of 768-dimensional vectors can exceed gRPC's default message size.

Run it:

```bash
uv run ingest.py --manifest manifest.json
```

**What to expect:**

```text
embedding N/N (voice)
N vectors embedded
N points in 'field_search'
```

The counter updates in place while it runs. Video and audio are the slowest records, and on CPU a large corpus takes a long time, so start with a few assets to confirm the run end to end before you embed everything. The vector count is higher than the record count, because long manual pages split into several chunks. The point count must equal the vector count.

If you change the corpus and run again, add `--recreate` so points left over from the old run are removed:

```bash
uv run ingest.py --manifest manifest.json --recreate
```

`admin.py` is a small helper for checking and dropping collections, which you need in steps 7 and 10:

**File: `admin.py`**

```python
"""Small helper: count or drop a collection.

  uv run admin.py count field_search
  uv run admin.py drop  field_search
"""
from __future__ import annotations

import sys

from actian_vectorai import VectorAIClient

action, name = sys.argv[1], sys.argv[2]
with VectorAIClient("localhost:6574") as client:
    if action == "count":
        print(f"{client.points.count(name)} points in '{name}'")
    elif action == "drop":
        client.collections.delete(name)
        print(f"dropped '{name}'")
    else:
        raise SystemExit("usage: admin.py count|drop <collection>")
```

The persistence check from the article's FAISS section uses it. Restart the container and count the points again:

```bash
docker compose restart
uv run admin.py count field_search
```

**What to expect:** the same point count as before the restart, with nothing re-embedded. That is the "Persistence across a restart" row of the FAISS table.

---

## Step 8: Write the Labelled Query Set

The query set is the part of the experiment you write by hand, and every hit rate in the article is only as good as these labels. Each query names its modality, the query itself (text, or the path to a photo or voice clip), the modality it should retrieve, and the list of `asset_id`s that count as a correct answer.

**File: `queries.example.json`**

```json
[
 {
  "query_modality": "text",
  "query": "how do I replace the impeller seal",
  "target_modality": "manual_text",
  "relevant": [
   "pump-a12"
  ]
 },
 {
  "query_modality": "photo",
  "query": "queries/photos/leaking_seal_01.jpg",
  "target_modality": "manual_page",
  "relevant": [
   "pump-a12"
  ]
 },
 {
  "query_modality": "text",
  "query": "exploded view of the impeller assembly",
  "target_modality": "manual_page",
  "relevant": [
   "pump-a12"
  ]
 },
 {
  "query_modality": "text",
  "query": "video showing how to bleed the hydraulic line",
  "target_modality": "video",
  "relevant": [
   "press-h7"
  ]
 },
 {
  "query_modality": "voice",
  "query": "queries/voice/grinding_noise.wav",
  "target_modality": "manual_text",
  "relevant": [
   "pump-a12"
  ]
 }
]
```

Copy it to `queries.json` and replace the examples with your own:

```bash
cp queries.example.json queries.json
mkdir -p queries/photos queries/voice
```

Follow these rules, because the article reports the numbers as measured:

- Cover the four pairings in the article's table: text to `manual_text`, photo to `manual_page`, text to `video`, and voice to `manual_text`. Add text to `manual_page` queries too, because step 11 uses them.
- Write at least 20 queries per pairing. With fewer, a single query moves the rate by more than five points.
- Query photos and voice clips must be new captures, not files already in `data/`. A photo that is also in the corpus retrieves itself and inflates the score. Put them in `queries/`.
- Label each query before you run anything, and have a second person check the labels without seeing the model's results.

---

## Step 9: Measure Hit Rate per Modality, Then the Hard Case

`eval_hitrate.py` embeds each query in its own modality, searches `field_search` with a filter on the target modality, and counts a hit when any of the top k results belongs to an asset in `relevant`. It prints one line per pairing and writes the same numbers to `hitrate.csv`.

**File: `eval_hitrate.py`**

```python
"""Hit rate at k per (query modality -> target modality) pair.

queries.json is a list of records:
  {"query_modality": "text"|"photo"|"voice", "query": "<text or file path>",
   "target_modality": "manual_text"|"manual_page"|"video", "relevant": ["pump-a12", ...]}
A query is a hit when any of the top-k results has an asset_id in "relevant".
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from actian_vectorai import Field, FilterBuilder, VectorAIClient

from embed import embed_media, embed_query, load_model


def embed_for(model, rec):
    qm = rec["query_modality"]
    if qm == "text":
        return embed_query(model, rec["query"])
    return embed_media(model, {"photo": "image", "voice": "audio"}[qm], rec["query"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--collection", default="field_search")
    ap.add_argument("--url", default="localhost:6574")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="hitrate.csv")
    args = ap.parse_args()

    queries = json.loads(Path(args.queries).read_text())
    model = load_model()
    tally = defaultdict(lambda: [0, 0])  # pair -> [hits, total]

    with VectorAIClient(args.url) as client:
        for rec in queries:
            flt = FilterBuilder().must(Field("modality").eq(rec["target_modality"])).build()
            hits = client.points.search(
                args.collection, embed_for(model, rec).tolist(),
                limit=args.k, filter=flt, with_payload=True,
            )
            ok = any(h.payload["asset_id"] in rec["relevant"] for h in hits)
            key = (rec["query_modality"], rec["target_modality"])
            tally[key][0] += int(ok)
            tally[key][1] += 1

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_modality", "target_modality", f"hit_rate_at_{args.k}", "queries"])
        for (qm, tm), (hit, total) in sorted(tally.items()):
            w.writerow([qm, tm, f"{hit / total:.3f}", total])
            print(f"{qm:>6} -> {tm:<12} hit@{args.k} = {hit / total:.3f}  ({hit}/{total})")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run eval_hitrate.py --queries queries.json --k 5
```

**What to expect:**

```text
 photo -> manual_page  hit@5 = x.xxx  (N/N)
  text -> manual_page  hit@5 = x.xxx  (N/N)
  text -> manual_text  hit@5 = x.xxx  (N/N)
  text -> video        hit@5 = x.xxx  (N/N)
 voice -> manual_text  hit@5 = x.xxx  (N/N)
```

The rates and counts go into the article's "Hit rate at 5" column, with the count in the Notes column. Keep `hitrate.csv` for Alex's sign-off.

`hard_case.py` re-runs only the photo to `manual_page` queries and prints the top five results for every miss, so you can sort each miss into "different part, similar silhouette" or "right assembly, wrong page". It also prints the range of top-1 scores for hits and for misses.

**File: `hard_case.py`**

```python
"""Print every photo -> manual_page query that missed, with its top results and scores.

Sort each miss by hand into: (a) different part, similar silhouette, or
(b) right assembly, wrong page. Compare the scores of misses with the scores
printed for hits to see whether a score threshold could separate them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from actian_vectorai import Field, FilterBuilder, VectorAIClient

from embed import embed_media, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--collection", default="field_search")
    ap.add_argument("--url", default="localhost:6574")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    photo_queries = [q for q in json.loads(Path(args.queries).read_text())
                     if q["query_modality"] == "photo" and q["target_modality"] == "manual_page"]
    model = load_model()
    flt = FilterBuilder().must(Field("modality").eq("manual_page")).build()

    hit_scores, miss_scores, misses = [], [], 0
    with VectorAIClient(args.url) as client:
        for rec in photo_queries:
            vec = embed_media(model, "image", rec["query"]).tolist()
            hits = client.points.search(args.collection, vec, limit=args.k, filter=flt, with_payload=True)
            if any(h.payload["asset_id"] in rec["relevant"] for h in hits):
                hit_scores.append(hits[0].score)
                continue
            misses += 1
            miss_scores.append(hits[0].score if hits else float("nan"))
            print(f"\nMISS {rec['query']}  expected {rec['relevant']}")
            for h in hits:
                print(f"  {h.score:.4f}  {h.payload['asset_id']:<14} {h.payload.get('path', '')}")

    n = len(photo_queries)
    print(f"\n{misses} misses out of {n} photo -> manual_page queries")
    if hit_scores and miss_scores:
        print(f"top-1 score on hits:   min {min(hit_scores):.4f}  max {max(hit_scores):.4f}")
        print(f"top-1 score on misses: min {min(miss_scores):.4f}  max {max(miss_scores):.4f}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run hard_case.py --queries queries.json
```

**What to expect:**

```text
MISS queries/photos/leaking_seal_03.jpg  expected ['pump-a12']
  x.xxxx  valve-b07      data/valve-b07/pages/service_manual_p012.png
  ...

N misses out of N photo -> manual_page queries
top-1 score on hits:   min x.xxxx  max x.xxxx
top-1 score on misses: min x.xxxx  max x.xxxx
```

Open the page images the misses returned and compare them with the query photo. If the two score ranges overlap, a score threshold cannot separate right answers from wrong ones, and the article's "Hard Case" section should recommend the confirmation step. If the miss scores sit clearly below the hit scores, the section can recommend a threshold just above the highest miss score.

---

## Step 10: Compare FAISS and VectorAI DB Under a Filter

This step needs at least 5,000 real EmbeddingGemma 2 vectors, which is far more than a small multimodal corpus produces. If `filter_compare.py` prints `skip 1000: only 7 vectors available`, the commands are right and the vector file is too small. `filter_corpus.py` builds the larger set from text you already have.

### 10a. Build 5,000 Vectors for the Test

`filter_corpus.py` reads every PDF and `.txt` file under `--source`, splits the text into 60-word chunks, and embeds them with the text-only model. The text-only model has 270M parameters, so it runs faster on a CPU, and its vectors sit in the same space as the full model's. You need about 300,000 words of text for 5,000 chunks, which is roughly two or three long service manuals. Any manuals or technical PDFs you have the rights to will do, because this test measures filtering, not answer quality. Put extra PDFs in a folder such as `filter_text/` if you do not want them in the search corpus.

**File: `filter_corpus.py`**

```python
"""Build a larger set of real EmbeddingGemma 2 vectors for filter_compare.py.

The filter test needs at least as many vectors as the largest --sizes value
(5,000 for the article). It reads every PDF and .txt file under --source, splits
the text into short chunks, embeds them with the text-only model (270M, faster
on CPU, same vector space), and writes filter_embeddings.npy.

  uv run filter_corpus.py --source data --target 200     # quick test first
  uv run filter_corpus.py --source data --target 5000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pymupdf

from embed import load_model


def texts(source: Path):
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() == ".pdf":
            with pymupdf.open(path) as doc:
                for page in doc:
                    yield page.get_text()
        elif path.suffix.lower() == ".txt":
            yield path.read_text(errors="ignore")


def chunks(source: Path, words: int):
    for text in texts(source):
        w = text.split()
        for i in range(0, len(w), words):
            piece = w[i : i + words]
            if len(piece) >= words // 2:        # skip page-end fragments
                yield " ".join(piece)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data")
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--words", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="filter_embeddings.npy")
    args = ap.parse_args()

    pieces = []
    for c in chunks(Path(args.source), args.words):
        pieces.append(c)
        if len(pieces) == args.target:
            break
    if len(pieces) < args.target:
        raise SystemExit(f"Only {len(pieces)} chunks of {args.words} words under {args.source}. "
                         f"Add more PDFs or text, or lower --words.")
    print(f"{len(pieces)} chunks ready")

    model = load_model(text_only=True)
    out, t0 = [], time.perf_counter()
    for i in range(0, len(pieces), args.batch):
        vecs = model.encode(pieces[i : i + args.batch], prompt_name="Document",
                            normalize_embeddings=True)
        if not np.isfinite(vecs).all():
            raise SystemExit("NaN in embeddings. Check the dtype.")
        out.append(vecs)
        done = min(i + args.batch, len(pieces))
        rate = done / (time.perf_counter() - t0)
        print(f"\r{done}/{len(pieces)} embedded | {rate:.1f}/s | ~{(len(pieces) - done) / rate / 60:.0f} min left",
              end="", flush=True)
    np.save(args.out, np.concatenate(out).astype("float32"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
```

Run a small test first, then the full set:

```bash
uv run filter_corpus.py --source filter_text --target 200
uv run filter_corpus.py --source filter_text --target 5000
```

**What to expect:**

```text
5000 chunks ready
5000/5000 embedded | x.x/s | ~0 min left
wrote filter_embeddings.npy
```

The counter shows the rate and an estimate of the minutes left. Note the rate from the 200-vector run, because 5,000 takes 25 times as long. If the script stops with `Only N chunks of 60 words`, add more PDFs or lower `--words`.

### 10b. Check That the Filter Works

`filter_check.py` builds a 200-point collection where 10 points are tagged `asset_id: "target"`, searches with a filter on that field and on a boolean field, and lists every returned point that breaks the filter. Run it once before the comparison, because a filter the server does not apply makes every VectorAI DB recall number meaningless.

**File: `filter_check.py`**

```python
"""Check that VectorAI DB applies a payload filter before trusting filter_compare.py.

Creates a 200-point collection where 10 points have asset_id "target", searches with
a filter on that field, and reports every returned point that breaks the filter.
"""
from __future__ import annotations

import numpy as np
from actian_vectorai import (
    Distance, Field, FilterBuilder, PointStruct, VectorAIClient, VectorParams,
)

NAME = "filter_check"
rng = np.random.default_rng(0)
vecs = rng.standard_normal((200, 768)).astype("float32")
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
targets = set(range(0, 200, 20))   # 10 targets

with VectorAIClient("localhost:6574") as client:
    if client.collections.exists(NAME):
        client.collections.delete(NAME)
    client.collections.create(NAME, vectors_config=VectorParams(size=768, distance=Distance.Cosine))
    client.points.upsert(NAME, points=[
        PointStruct(id=i, vector=vecs[i].tolist(),
                    payload={"asset_id": "target" if i in targets else "other", "flag": i in targets})
        for i in range(200)])
    client.vde.flush(NAME)
    client.vde.wait_for_ready(NAME)

    for label, flt in [("keyword asset_id == 'target'", FilterBuilder().must(Field("asset_id").eq("target")).build()),
                       ("boolean flag == True", FilterBuilder().must(Field("flag").eq(True)).build())]:
        hits = client.points.search(NAME, vecs[1].tolist(), limit=10, filter=flt, with_payload=True)
        ids = [int(h.id) for h in hits]
        off = [i for i in ids if i not in targets]
        print(f"{label:<30} returned {len(ids)} ids {ids}")
        print(f"{'':<30} off-filter: {off if off else 'none'}")
    client.collections.delete(NAME)
```

```bash
uv run filter_check.py
```

**What to expect:** both lines return 10 IDs, all multiples of 20, and `off-filter: none`. If either line lists off-filter IDs, stop and send that output to the Actian team or the Discord before running the comparison.

### 10c. Run the Comparison

`filter_compare.py` samples each corpus size from `filter_embeddings.npy`, tags a random 1% of it as the target asset, and runs 100 queries through a FAISS HNSW index with post-filtering, the same index with an `IDSelector`, and VectorAI DB with a payload filter. Before it searches VectorAI DB, it flushes the collection and waits until indexing finishes, and it reads the index size from the database's own statistics. Every line also shows how many of the 10 results came back and how many broke the filter.

**File: `filter_compare.py`**

```python
"""FAISS HNSW against VectorAI DB under a filter that admits a fixed share of the corpus.

For each corpus size, the script samples that many vectors from --vectors, tags a
random --filter-share of them with asset_id "target" (standing in for one asset)
and the rest with "other", and runs the same filtered queries three ways:

  faiss_post_filter   HNSW search for the top k, then drop results outside the target
  faiss_id_selector   HNSW search with an IDSelector built from your own metadata table
  vectorai            VectorAI DB with a payload filter on asset_id inside the search

Ground truth is exact filtered search in NumPy. Each line also reports how many
results came back on average and how many of them broke the filter, so a filter the
server did not apply shows up in the output instead of as a low recall number.
"""
from __future__ import annotations

import argparse
import time

import faiss
import numpy as np
from actian_vectorai import (
    Distance, Field, FilterBuilder, PointStruct, SearchParams, VectorAIClient, VectorParams,
)

SCRIPT_VERSION = "filter_compare v3 (HNSW, keyword filter, flush + stats)"


def p50_ms(times: list[float]) -> float:
    return float(np.percentile(times, 50) * 1000)


def recall(found: set, gt: set) -> float:
    return len(found & gt) / max(len(gt), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="filter_embeddings.npy")
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--filter-share", type=float, default=0.01)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--ef-search", type=int, default=16, help="FAISS efSearch (FAISS default is 16)")
    ap.add_argument("--vai-ef", type=int, default=None, help="optional hnsw_ef for VectorAI DB searches")
    ap.add_argument("--url", default="localhost:6574")
    ap.add_argument("--hardware", required=True, help='e.g. "AMD Ryzen 5 2500U, 8 GB RAM, no CUDA GPU, float32"')
    args = ap.parse_args()

    vecs = np.load(args.vectors).astype("float32")
    rng = np.random.default_rng(0)
    print(SCRIPT_VERSION)
    print(f"Hardware: {args.hardware}")
    print(f"Vectors: {len(vecs)} from {args.vectors} | filter admits {args.filter_share:.0%} | k={args.k} | "
          f"FAISS HNSW M={args.m}, efSearch={args.ef_search} | VectorAI DB hnsw_ef={args.vai_ef or 'server default'}\n")

    search_params = SearchParams(hnsw_ef=args.vai_ef) if args.vai_ef else None

    with VectorAIClient(args.url) as client:
        for size in args.sizes:
            if size > len(vecs):
                print(f"skip {size}: only {len(vecs)} vectors available")
                continue
            sub = vecs[rng.choice(len(vecs), size, replace=False)]
            n_target = max(args.k, round(size * args.filter_share))
            target = np.sort(rng.choice(size, n_target, replace=False)).astype("int64")
            target_set = set(target.tolist())
            qs = sub[rng.choice(size, args.queries, replace=False)]
            truth = [set(target[np.argsort(-(sub[target] @ q))][: args.k].tolist()) for q in qs]
            rows = {}

            index = faiss.IndexHNSWFlat(sub.shape[1], args.m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = args.ef_search
            index.add(sub)
            faiss_mb = faiss.serialize_index(index).nbytes / 1e6

            t, r, n, bad = [], [], [], 0
            for q, gt in zip(qs, truth):
                s = time.perf_counter()
                _, I = index.search(q[None], args.k)
                t.append(time.perf_counter() - s)
                kept = {int(i) for i in I[0] if int(i) in target_set}
                r.append(recall(kept, gt)); n.append(len(kept))
            rows["faiss_post_filter"] = (np.mean(r), p50_ms(t), faiss_mb, np.mean(n), bad)

            params = faiss.SearchParametersHNSW(sel=faiss.IDSelectorBatch(target), efSearch=args.ef_search)
            t, r, n, bad = [], [], [], 0
            for q, gt in zip(qs, truth):
                s = time.perf_counter()
                _, I = index.search(q[None], args.k, params=params)
                t.append(time.perf_counter() - s)
                got = {int(i) for i in I[0] if i >= 0}
                bad += len(got - target_set)
                r.append(recall(got, gt)); n.append(len(got))
            rows["faiss_id_selector"] = (np.mean(r), p50_ms(t), faiss_mb, np.mean(n), bad)

            name = f"filter_compare_{size}"
            if client.collections.exists(name):
                client.collections.delete(name)
            client.collections.create(name, vectors_config=VectorParams(size=sub.shape[1], distance=Distance.Cosine))
            points = [PointStruct(id=i, vector=sub[i].tolist(),
                                  payload={"idx": i, "asset_id": "target" if i in target_set else "other"})
                      for i in range(size)]
            for start in range(0, len(points), 256):
                client.points.upsert(name, points=points[start : start + 256])
            client.vde.flush(name)
            client.vde.wait_for_ready(name)          # do not search a collection that is still indexing
            stats = client.vde.get_stats(name)
            vai_mb = (stats.storage_bytes + stats.index_memory_bytes) / 1e6

            flt = FilterBuilder().must(Field("asset_id").eq("target")).build()
            t, r, n, bad = [], [], [], 0
            for q, gt in zip(qs, truth):
                s = time.perf_counter()
                hits = client.points.search(name, q.tolist(), limit=args.k, filter=flt,
                                            params=search_params, with_payload=True)
                t.append(time.perf_counter() - s)
                got = {int(h.payload["idx"]) for h in hits}
                bad += sum(1 for h in hits if h.payload.get("asset_id") != "target")
                r.append(recall(got, gt)); n.append(len(hits))
            rows["vectorai"] = (np.mean(r), p50_ms(t), vai_mb, np.mean(n), bad)
            client.collections.delete(name)

            print(f"corpus {size}  filter admits {n_target} vectors ({n_target / size:.1%})  "
                  f"VectorAI DB indexed {stats.indexed_vectors}/{stats.total_vectors}")
            for label, (rec, ms, mb, avg_n, viol) in rows.items():
                print(f"  {label:<18} recall@{args.k} {rec:.3f}   p50 {ms:.3f} ms   index {mb:.1f} MB   "
                      f"returned {avg_n:.1f}/{args.k}   off-filter {viol}")


if __name__ == "__main__":
    main()
```

The Community Edition caps you at 5,000 vectors, so drop `field_search` for this step and restore it from the cache afterwards:

```bash
uv run admin.py drop field_search
uv run filter_compare.py --sizes 1000 2500 5000 --filter-share 0.01 \
      --hardware "AMD Ryzen 5 2500U, 8 GB RAM, no CUDA GPU, float32"
uv run ingest.py --from-cache
```

**What to expect:**

```text
filter_compare v3 (HNSW, keyword filter, flush + stats)
Hardware: AMD Ryzen 5 2500U, 8 GB RAM, no CUDA GPU, float32
Vectors: 5000 from filter_embeddings.npy | filter admits 1% | k=10 | FAISS HNSW M=32, efSearch=16 | VectorAI DB hnsw_ef=server default

corpus 1000  filter admits 10 vectors (1.0%)  VectorAI DB indexed 1000/1000
  faiss_post_filter  recall@10 x.xxx   p50 x.xxx ms   index x.x MB   returned x.x/10   off-filter 0
  faiss_id_selector  recall@10 x.xxx   p50 x.xxx ms   index x.x MB   returned x.x/10   off-filter 0
  vectorai           recall@10 x.xxx   p50 x.xxx ms   index x.x MB   returned x.x/10   off-filter 0
corpus 2500  ...
corpus 5000  ...
```

The first line must read `filter_compare v3`. If it does not, you are running an older copy of the script. Three checks decide whether the numbers can go in the article:

- `off-filter` is 0 on every line. Anything else means the filter was not applied.
- `indexed` equals the corpus size, so the search ran against a finished index.
- `returned` explains the recall. Post-filtering usually returns fewer than 10 results, because most of its top 10 fall outside the filter.

If `faiss_id_selector` recall is low, run again with `--ef-search 64` and then `--ef-search 256`, and report the recall and latency at each setting. If `vectorai` recall is low with `off-filter 0`, run again with `--vai-ef 64` and `--vai-ef 256` to see whether the server's search depth explains it.

---

## Step 11: Compare ColPali and EmbeddingGemma 2 on Manual Pages (GPU)

`colpali_compare.py` embeds every page in `pages.json` with ColPali and with EmbeddingGemma 2, runs the text to `manual_page` queries from `queries.json` against both, and prints hit rate at k, bytes per page, and indexing time. ColPali scores pages on the client with its own MaxSim function, as the article states.

**File: `colpali_compare.py`**

```python
"""ColPali against EmbeddingGemma 2 on manual pages only.

Needs a GPU and the colpali extra:  uv sync --extra colpali
Pages are scored client-side with ColPali's MaxSim (processor.score_multi_vector).
The SDK also exposes MultiVectorConfig with a MaxSim comparator; this script does not
rely on it, because it has not been run against a live server.

pages.json:   [{"asset_id": "pump-a12", "path": "manuals/pump-a12/p03.png"}, ...]
queries.json: same schema as eval_hitrate.py, restricted to target_modality "manual_page".
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

from embed import embed_media, embed_query, load_model

COLPALI_ID = "vidore/colpali-v1.3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="pages.json")
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    pages = json.loads(Path(args.pages).read_text())
    queries = [q for q in json.loads(Path(args.queries).read_text())
               if q["query_modality"] == "text" and q["target_modality"] == "manual_page"]

    if torch.cuda.is_available():
        device, dtype = "cuda:0", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32   # works, but slow for a 3B model
    cp = ColPali.from_pretrained(COLPALI_ID, torch_dtype=dtype, device_map=device).eval()
    proc = ColPaliProcessor.from_pretrained(COLPALI_ID)

    t0 = time.perf_counter()
    page_embs = []
    for i in range(0, len(pages), 4):
        batch = proc.process_images([Image.open(p["path"]).convert("RGB") for p in pages[i : i + 4]]).to(cp.device)
        with torch.no_grad():
            page_embs.extend(list(cp(**batch).cpu()))
    cp_index_s = time.perf_counter() - t0
    bytes_per_page = page_embs[0].numel() * page_embs[0].element_size()

    gemma = load_model()
    t0 = time.perf_counter()
    gem_page = np.stack([embed_media(gemma, "image", p["path"]) for p in pages])
    gm_index_s = time.perf_counter() - t0
    gem_bytes = gem_page.shape[1] * 2  # bf16

    hits_cp = hits_gm = 0
    for q in queries:
        qb = proc.process_queries([q["query"]]).to(cp.device)
        with torch.no_grad():
            qe = cp(**qb)
        scores = proc.score_multi_vector(qe, page_embs)[0]
        top_cp = scores.topk(min(args.k, len(pages))).indices.tolist()
        top_gm = np.argsort(-(gem_page @ embed_query(gemma, q["query"])))[: args.k].tolist()
        hits_cp += any(pages[i]["asset_id"] in q["relevant"] for i in top_cp)
        hits_gm += any(pages[i]["asset_id"] in q["relevant"] for i in top_gm)

    n = len(queries)
    print(f"queries: {n}")
    print(f"EmbeddingGemma 2  hit@{args.k} {hits_gm / n:.3f}   {gem_bytes} bytes/page (bf16, one vector)   indexed in {gm_index_s:.1f}s")
    print(f"ColPali           hit@{args.k} {hits_cp / n:.3f}   {bytes_per_page} bytes/page "
          f"({page_embs[0].shape[0]} vectors x {page_embs[0].shape[1]})   indexed in {cp_index_s:.1f}s")


if __name__ == "__main__":
    main()
```

Install the extra and run it:

```bash
uv sync --extra colpali
uv run hf download vidore/colpali-v1.3
uv run colpali_compare.py --pages pages.json --queries queries.json --k 5
```

**What to expect:**

```text
queries: N
EmbeddingGemma 2  hit@5 x.xxx   1536 bytes/page (bf16, one vector)   indexed in x.xs
ColPali           hit@5 x.xxx   N bytes/page (N vectors x 128)   indexed in x.xs
```

The EmbeddingGemma 2 line always shows 1,536 bytes, because one 768-dimensional bf16 vector is 768 × 2 bytes. ColPali's line shows the measured size of its first page, which should land near the 263,680 bytes the article calculates from the model card. If it differs, use the measured number in the article. ColPali is a 3B-parameter model, so on CPU this step runs very slowly. Run it on the GPU machine and report the hardware with the table.

---

## Step 12: Map the Output to the Article

| Article placeholder | Comes from |
|---|---|
| Hit-rate table and the test-section `[RESULT]` paragraphs | `eval_hitrate.py` output and `hitrate.csv` |
| Hard-case `[RESULT]` paragraphs | `hard_case.py` miss list and score ranges, sorted by hand |
| FAISS table rows for recall, latency and index size at each size | `filter_compare.py` output (step 10b) |
| `IDSelector` paragraph | the `faiss_id_selector` lines |
| "Persistence across a restart" row | the restart check in step 7 |
| ColPali table | `colpali_compare.py` output |
| Hardware line | the `--hardware` string and the dtype from step 5 |
| Model commit hash | the snapshot path printed in step 4 |
| Image digest | the `docker image inspect` output in step 3 |
| Query count and labellers | your `queries.json` and the people who checked it |

Keep CPU and GPU runs in separate tables, and send the CSV and raw terminal output to Alex with the draft so every number can be traced to a run.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Client is not connected` or connection refused on port 6574 | The container is not running | `docker compose ps`, then `docker compose up -d` |
| `ValueError: Embedding contains NaN or inf` | The model loaded in float16 | Do not change `load_model`. Check the dtype line from step 5 |
| 401 or 403 from Hugging Face | Not signed in, or the model terms not accepted | `uv run hf auth login`, then accept the terms on the model page |
| `KeyError` on a modality in `ingest.py` | A manifest record has a modality name outside the five supported ones | Rebuild with `make_manifest.py` rather than editing the manifest by hand |
| `... over the 327s limit. Split it first.` | A voice note is too long | Split it with `ffmpeg -f segment -segment_time 300` |
| Video fails to decode | ffmpeg missing, or the file is not MP4 | Install ffmpeg, convert with `ffmpeg -i in.mov out.mp4` |
| `skip 5000: only N vectors available` | `filter_embeddings.npy` has fewer vectors than the requested size | Run `filter_corpus.py --target 5000` (step 10a) |
| Errors about the vector limit while upserting | The 5,000-vector Community Edition cap | Drop `field_search` before step 10, as shown there |
| Very slow embedding | Running in float32 on CPU | Expected. Test with a few assets first, or use a GPU |
| Hit rates near 1.000 everywhere | Query photos or clips are also in the corpus, or `relevant` lists every asset | Use fresh captures in `queries/` and label precisely |