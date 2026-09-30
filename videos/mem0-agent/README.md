# Persistent agent memory with mem0 and VectorAI DB

A small agent that remembers what you tell it across separate processes. [mem0](https://github.com/mem0ai/mem0) decides what is worth remembering; [Actian VectorAI DB](https://hub.docker.com/r/actian/vectorai) stores the memories as vectors on your own machine. The LLM and the embedder both run locally through Ollama, so nothing needs an API key.

## How it works

mem0 picks its vector store from a fixed list of providers, and VectorAI DB isn't on it. The way in is mem0's `langchain` provider, which accepts a ready-made LangChain `VectorStore` object as its `client`. `vectorai_store.py` is that object: a thin LangChain `VectorStore` over the VectorAI DB client.

```
you ──► agent.py ──► mem0.search ──► VectorAIStore ──► VectorAI DB
            │                                              ▲
            ├──► Ollama (llama3.2) answers                 │
            └──► mem0.add ──► extracts facts ──► embeds ───┘
```

Each turn searches memory before the model answers, then hands the exchange to mem0, which extracts any facts worth keeping and writes them back.

## Prerequisites

- **Docker**, to run VectorAI DB
- **Python 3.10+** and [uv](https://docs.astral.sh/uv/)
- **[Ollama](https://ollama.com)** with two models pulled:

  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

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
python check.py                          # confirm the database is reachable
python seed.py                           # write a few memories, print what mem0 extracted
python query.py "which database"         # a fresh process: the memories are still there
python agent.py                          # chat loop; Ctrl-C to quit
python reset.py                          # drop the collection to start over
```

Expected output from `seed.py` then `query.py` looks like this (exact wording varies, because the LLM does the extraction):

```
  + User moved project off Chroma in March 2026 due to concurrency issues
  + User moved project to Postgres with pgvector in March 2026 due to concurrency issues
  + Assistant will keep answers short when replying to user
  ...

0.493  User moved project to Postgres with pgvector in March 2026 due to concurrency issues
0.457  Postgres with pgvector has been fine so far
```

## Files

| File | What it does |
| --- | --- |
| `vectorai_store.py` | LangChain `VectorStore` adapter over VectorAI DB. This is the integration. |
| `memory.py` | mem0 config: Ollama LLM and embedder, VectorAI DB as the store. Models and collection name live here. |
| `agent.py` | The turn loop: search memory, answer, add the turn to memory. |
| `seed.py` | Writes four example conversations for user `ada`. |
| `query.py` | Searches memory from a new process. |
| `check.py` / `reset.py` | Connection check and collection drop. |
| `quiet.py` | Disables mem0 telemetry and silences dependency warnings. Imported before mem0. |

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VECTORAI_HOST` | `localhost:6574` | VectorAI DB gRPC address |
| `MEM0_TELEMETRY` | `False` (set by `quiet.py`) | mem0's PostHog telemetry |

To change the models, edit `LLM_MODEL` and `EMBED_MODEL` in `memory.py`. If you change the embedder, set `EMBED_DIM` to its output width and run `reset.py`: the collection is created at that width, and a mismatch fails on every insert. (`nomic-embed-text` is 768; mem0's default OpenAI embedder is 1536.)

## Troubleshooting

- **`CollectionNotFoundError` after restarting the database.** VectorAI DB keeps collections on disk but doesn't serve them again until they're reopened. `VectorAIStore` calls `vde.open_collection()` on startup to handle this. If you write your own client code, do the same, and don't delete a collection that looks missing.
- **Port 6574 already in use.** Another VectorAI DB container is running. Reuse it, or map different ports and set `VECTORAI_HOST`.
- **Ollama errors (`model not found`).** Pull both models listed under Prerequisites.

## Verified

Tested on 2026-09-30 on Windows 11 with Python 3.13, a fresh `actian/vectorai:latest` container and Ollama: `check`, `reset`, `seed`, `query` and one `agent` turn all ran, and the memories survived a database restart.

## License

MIT. See [LICENSE](LICENSE).
