# Persistent Semantic Memory for Pydantic AI With VectorAI DB

A Pydantic AI agent that remembers facts across separate runs. Memories are embedded locally with sentence-transformers, stored in Actian VectorAI DB running in Docker, and retrieved by meaning rather than by matching words. The chat model runs on Together AI.

## How It Works

Each turn runs through the `ask()` function in `memory_agent.py`:

1. The store embeds the user's message and searches VectorAI DB for the closest memories that belong to that user.
2. The `assistant` agent answers, with any matching memories added to the prompt inside `<memory>` tags.
3. The `extractor` agent reads only the user's original message and returns the facts worth saving.
4. The store embeds each fact and saves it to VectorAI DB with its user ID, session ID, memory type, and timestamp.

The extractor never sees recalled memories or the assistant's reply, so it cannot re-save old notes or store the model's guesses.

## Project Files

| File | Purpose |
| --- | --- |
| `docker-compose.yml` | Runs VectorAI DB and persists its data in `./data`. |
| `vectoraidb_memory_store.py` | `VectorAIDBStore` with `store`, `search`, `list`, and `delete`. Run it directly to create the collection. |
| `memory_agent.py` | The `assistant` and `extractor` agents, and the `ask()` function. |
| `session_1.py` | Tells the agent two facts, then exits. |
| `session_2.py` | Starts as a new process and asks a question that needs those facts. |

## Prerequisites

- Docker, with at least 8 GB of RAM available for the VectorAI DB container
- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- A [Together AI](https://www.together.ai/) API key

## Quickstart

### 1. Start VectorAI DB

```bash
docker compose up -d
docker ps
```

The `vectorai_db` container should show as `Up`. The Python SDK connects to it on port 6574.

On Linux or WSL2, if the container exits right away, fix the data folder's owner and start it again:

```bash
sudo chown -R 999:999 ./data
docker compose up -d
```

### 2. Install the dependencies

```bash
uv venv
uv pip install pydantic-ai actian-vectorai-client sentence-transformers
```

If you don't use uv, create a virtual environment and run `pip install pydantic-ai actian-vectorai-client sentence-transformers` instead.

### 3. Download the embedding model

```bash
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

This downloads the model once and caches it locally. You don't need a Hugging Face account.

### 4. Create the memory collection

```bash
uv run vectoraidb_memory_store.py
```

You should see `Collection 'agent_memory' is ready`. The command is safe to run again.

### 5. Set your Together AI key

```bash
export TOGETHER_API_KEY=your-together-key
```

On Windows, use `set TOGETHER_API_KEY=your-together-key`.

### 6. Run both sessions

```bash
uv run session_1.py && uv run session_2.py
```

The `&&` starts session 2 only after session 1 finishes. Expect output like this:

```text
memory search 'For future chats: ...' -> 0 hits []
memory store  [fact] 'our production EKS cluster runs in eu-west-1'
memory store  [fact] 'we deploy infrastructure with Terraform through GitHub Actions'
agent: I've taken note of that for our future conversations. ...

memories on disk: 2
memory search 'What tools do I use for infrastructure deployments?' -> 1 hits [0.518]
agent: You use Terraform for infrastructure deployments, and GitHub Actions as the deployment pipeline tool.
```

Session 2 finds the relevant memory even though the question never mentions Terraform or GitHub Actions. Agent replies and scores vary slightly between runs.

## Configuration

| Setting | Where | Default |
| --- | --- | --- |
| Chat model | `TOGETHER_MODEL` environment variable | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| VectorAI DB address | `VectorAIDBStore(url=...)` | `localhost:6574` |
| Collection name | `VectorAIDBStore(collection=...)` | `agent_memory` |
| Embedding model | `VectorAIDBStore(model=...)` | `sentence-transformers/all-MiniLM-L6-v2` |
| Similarity threshold | `VectorAIDBStore(threshold=...)` | `0.3` |

Any Together AI model that supports function calling works as the chat model. If you change the embedding model, delete and recreate the collection, because VectorAI DB can't change a collection's vector size after creation.

## Using the Store in Your Own Code

```python
from memory_agent import ask
from vectoraidb_memory_store import VectorAIDBStore

with VectorAIDBStore() as store:
    reply = ask(store, user_id="user-42", session_id="abc123", message="Hello")

    store.list(user_id="user-42")                         # all of a user's memories
    store.delete(user_id="user-42", session_id="abc123")  # remove one session
    store.delete(user_id="user-42")                       # erase a user's memories
```

Every search, list, and delete filters on `user_id`, so one user never reads or removes another user's memories. In a real application, take `user_id` from your authentication layer.

## Reset

```bash
docker compose down && rm -rf data
docker compose up -d
uv run vectoraidb_memory_store.py
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Error mentioning `TOGETHER_API_KEY` | Set the key in the current terminal (step 5). |
| Connection refused on port 6574 | Start the container with `docker compose up -d` and check `docker logs vectorai_db`. |
| Session 2 shows `memories on disk: 0` | Run the sessions in order with `&&`, not in parallel. |
| Session 2 finds memories but reports `0 hits` | Lower the threshold, for example `VectorAIDBStore(threshold=0.2)`. |
| Hugging Face "unauthenticated requests" warning | Harmless. Run `export HF_HUB_OFFLINE=1` after step 3 to hide it. |
| Slow container on Apple Silicon | The compose file sets `platform: linux/amd64`, so Docker emulates x86. Expect slower startup. |
