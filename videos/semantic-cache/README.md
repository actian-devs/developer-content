# Semantic cache for LLM calls with FastAPI and VectorAI DB

A FastAPI service that sits in front of an LLM and skips the model call when it has already answered a question that means the same thing. Each incoming question is embedded and compared against past questions stored in [Actian VectorAI DB](https://hub.docker.com/r/actian/vectorai). If the nearest one is similar enough, the service returns its stored answer in milliseconds. Otherwise it calls the model and writes the new answer back to the cache.

```
POST /ask ──► embed question ──► search VectorAI DB ──► score ≥ THRESHOLD? ──yes──► cached answer
                                                              │
                                                              no
                                                              ▼
                                         call the LLM ──► return answer ──► store it (background)
```

## Prerequisites

- **Docker**, to run VectorAI DB
- **Python 3.10+** and [uv](https://docs.astral.sh/uv/)
- **An LLM endpoint.** By default the service calls [Ollama](https://ollama.com) with `llama3.2` (`ollama pull llama3.2`). To use your own model, replace `complete()` in `app/llm.py`.

Embeddings are computed locally with [fastembed](https://github.com/qdrant/fastembed) (`all-MiniLM-L6-v2`, downloaded on first run). No API keys are needed.

## Setup

Start VectorAI DB (gRPC on 6574, web UI on 6575):

```bash
./01_start_vectorai.sh
```

On Windows, or without bash, run the same command directly:

```bash
docker run -d --name vectorai -v vectorai_data:/var/lib/actian-vectorai -p 6573-6575:6573-6575 -e ACTIAN_VECTORAI_ACCEPT_EULA=YES actian/vectorai:latest
```

Create the environment and install:

```bash
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Run

```bash
python -m app.cache --init            # create the llm_cache collection
python tune.py                        # threshold sweep over the labelled pairs
uvicorn app.main:app                  # start the service on :8000
python bench.py                       # in a second terminal: 60 requests, hit rate and latency
python reset.py                       # empty the cache to start over (safe with the service running)
```

Send a single request:

```bash
curl -s localhost:8000/ask -H "content-type: application/json" -d "{\"question\": \"How do I reset my password?\"}"
```

```json
{"answer": "...", "cached": false, "ms": 1834.2}
```

Ask the same thing in different words and `cached` flips to `true`. You can also try the endpoint from the interactive docs at http://localhost:8000/docs.

## Choosing the threshold

`THRESHOLD` in `app/cache.py` is the design decision that matters. Set it too low and the cache confidently serves the answer to a different question ("cancel my order" vs "cancel my subscription"). Set it too high and nothing matches, so every request goes to the model.

`tune.py` embeds the pairs in `data/labelled_pairs.jsonl` (each marked as the same or a different intent) and prints, for each threshold, how many same-intent pairs would hit and how many different-intent pairs would falsely hit:

```
30 pairs | 15 same intent | 15 different

threshold   hit rate   false hits
   0.50         60%     10/15
   ...
   0.60         60%     8/15   <- THRESHOLD in app/cache.py
   ...
   0.70         33%     3/15
   ...
   0.86         13%     0/15
   ...
   0.92          0%     0/15
```

The shipped `0.60` favours hit rate: on the sample pairs it catches 60% of rephrasings but also lets 8 of 15 different-intent pairs through. For production traffic, start higher and work down. The sample pairs are only a starting point. Replace them with pairs from your own query logs before you trust the number, and re-tune when your traffic changes.

## What gets stored

Each cache entry is one vector plus a payload:

| Field | Why |
| --- | --- |
| `query`, `answer` | The original question and the answer served for it |
| `created_at`, `ttl` | Entries expire after 24 hours by default (`TTL_SECONDS`) |
| `model`, `prompt_version` | If you change the model or bump `PROMPT_VERSION`, old answers stop being served, so it's safe to deploy a change behind the cache |
| `scope` | Optional partition (for example a tenant or locale), passed as `"scope"` in the request body. An entry only matches requests with the same scope. |

## Files

| File | What it does |
| --- | --- |
| `app/cache.py` | `SemanticCache`: `lookup()` (the hot path) and `store()` (write-on-miss). All tuning constants live here. |
| `app/main.py` | FastAPI app with one route, `POST /ask`. Writes back in a background task after the response is sent. |
| `app/llm.py` | The model call the cache sits in front of. |
| `tune.py` | Threshold sweep over `data/labelled_pairs.jsonl`. |
| `bench.py` | Sends `data/batch.json` (60 questions, 12 intents) to the running service and reports hit rate and median latency. |
| `reset.py` | Drops and recreates the collection. |

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VECTORAI_HOST` | `localhost:6574` | VectorAI DB gRPC address |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `LLM_MODEL` | `llama3.2` | Model name. Also stamped on every cache entry. |

## Troubleshooting

- **`CollectionNotFoundError` after restarting the database.** VectorAI DB keeps collections on disk but doesn't serve them again until they're reopened. `SemanticCache` calls `vde.open_collection()` on startup to handle this. If you write your own client code, do the same, and don't delete a collection that looks missing.
- **Port 6574 already in use.** Another VectorAI DB container is running. Reuse it, or map different ports and set `VECTORAI_HOST`.
- **`bench.py` shows only misses.** Check that `uvicorn` is running, and that `THRESHOLD` isn't set above what `tune.py` shows for your pairs.

## Verified

Tested on 2026-09-30 on Windows 11 with Python 3.13, a fresh `actian/vectorai:latest` container and Ollama `llama3.2`:

- `tune.py` output as shown above
- A rephrased password question: miss at 17.4 s, then a hit at 61 ms
- `bench.py`: 33 hits and 27 misses; median 155 ms for a hit vs 7.4 s for a miss
- The cache still served hits after a database restart

## License

MIT. See [LICENSE](LICENSE).
