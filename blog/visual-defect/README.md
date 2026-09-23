# Visual Defect Search

A CLIP-powered visual defect search demo using Actian VectorAI. Images are embedded with OpenAI CLIP and stored in a VectorAI collection for text and image similarity queries.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop with WSL integration enabled
- MVTec AD downloaded and extracted locally

## Run

Start VectorAI:

```bash
docker compose up -d
```

Install the Python environment:

```bash
uv sync
```

Verify the VectorAI connection:

```bash
uv run smoke_test.py
```

Index the MVTec AD images. The dataset should contain paths such as `bottle/test/scratch/*.png`:

```bash
uv run ingest.py --data ./mvtec_ad
```

Search the indexed images:

```bash
uv run query.py --text "scratched metal"
uv run query.py --image ./path/to/image.png
```

The VectorAI service uses ports `6573` through `6575`. Local VectorAI data is stored in `vectorai_data/` and is intentionally not committed.
