# Astra Codebase Memory

This example gives an Astra coding agent local, cross-session memory through Actian VectorAI DB. Session 1 repairs one environment-configuration normalization bug and stores the tested fix. Session 2 starts a separate Responses API chain, searches the same collection, and applies the retrieved lesson to a different configuration field. Correctness is verified separately with fresh copies of the original untouched acceptance tests.

The default command is an offline preview driven by deterministic fake Responses. It makes no OpenAI request. Live mode is opt-in, paid, and protected by explicit flags, Standard processing, an append-only local usage ledger, and a per-request budget check.

## Included files

| Path | Purpose |
| --- | --- |
| `video_demo.py` | Video-friendly two-session entry point with concise and verbose output. |
| `astra_agent.py` | Raw Responses API orchestration and tool-call safety gates. |
| `vectoraidb_memory_tools.py` | In-memory and Actian VectorAI DB memory backends plus local embeddings. |
| `coding_tools.py` | Project-root-confined file, edit, and test tools. |
| `cost_guard.py` | Worst-case request authorization and append-only usage ledger. |
| `session_demo.py`, `demo_support.py`, `settings.py` | Shared transport, workspace, evidence, configuration, and demo logic. |
| `demo_projects/templates/` | Immutable Scenario Alpha and Beta fixtures and acceptance tests. |

## Prerequisites

- Python 3.12
- Docker Desktop or Docker Engine with Compose
- About 8 GB of RAM available for VectorAI DB
- An OpenAI API account with access to `gpt-6-astra`, only for the optional paid live run

## Install

From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
cp .env.example .env
```

PowerShell equivalents for activation and environment setup are:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
Copy-Item .env.example .env
```

`requirements.txt` deliberately selects the CPU-only PyTorch index. The first run downloads `sentence-transformers/all-MiniLM-L6-v2` into the ignored `.cache/` directory and uses its normalized 384-dimensional embeddings on CPU.

## Offline preview

The offline preview uses fake Responses and an in-memory backend. Docker and an API key are not required.

```bash
python -u video_demo.py --run-id UNIQUE-OFFLINE-ID
```

Add `--verbose` to show detailed recorded tool events:

```bash
python -u video_demo.py --run-id UNIQUE-OFFLINE-ID --verbose
```

Every run ID must be new. Disposable workspaces are created below the ignored `runs/` directory and cleaned after a successful run.

## Paid live demonstration

The live path uses `gpt-6-astra` through the Responses API and incurs API charges. Before running it:

1. Start the isolated VectorAI DB service and wait for it to become healthy.
2. Ensure the `astra_codebase_memory` collection exists, is open and searchable, and contains zero records.
3. Put `OPENAI_API_KEY` in `.env` without printing it.
4. Keep `ASTRA_PROCESSING_TIER=standard`.
5. Choose a fresh run ID and explicitly approve live cost.

```bash
docker compose up -d
docker compose ps

python -u video_demo.py \
  --run-id UNIQUE-LIVE-ID \
  --live \
  --approve-live-cost
```

Live mode cannot run unless both flags are present and the API key is available. The agent must search memory before editing, and it may store a fix only after its test command passes. Session 2 receives no `previous_response_id` from Session 1. Retrieval proves that the earlier fix was available across response chains; the fresh untouched acceptance-test audit is the proof that each application repair works. Agent edits to test files are excluded from that audit.

Local run data, private transcripts, and `evidence/usage.jsonl` are ignored by Git. The demo cleanup deletes only memory IDs and workspaces created by its own successful run. It does not delete unrelated database records.

## Verify the package

```bash
python -m pip check
python -m ruff check .
python -m ruff format --check .
python -m compileall -q astra_agent.py coding_tools.py cost_guard.py demo_support.py session_demo.py settings.py vectoraidb_memory_tools.py video_demo.py demo_projects/templates
docker compose config --quiet
```

To confirm the fixtures begin in their intended failing state:

```bash
python -m pytest -q demo_projects/templates/scenario_alpha/tests
python -m pytest -q demo_projects/templates/scenario_beta/tests
```

Each command should report two failures and two passes before the agent runs.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| Ports `16573` to `16575` are occupied | Stop and inspect the owner. Do not reuse or modify another project's database. Change both Compose mappings and the matching `.env` URLs only for a new isolated deployment. |
| VectorAI DB is unhealthy | Run `docker compose ps` and `docker compose logs vectorai`. The demo stops rather than creating or replacing unexpected storage. |
| Live preflight says the collection is not empty | Review the records and use a separate project-owned database. The live demo intentionally refuses to delete unknown data. |
| Hugging Face model download fails | Check network access, then rerun installation or the offline preview. After the model is cached, `HF_HUB_OFFLINE=1` prevents network lookup. |
| Live mode refuses to start | Confirm both live flags, a nonempty `OPENAI_API_KEY`, a fresh run ID, and `ASTRA_PROCESSING_TIER=standard`. |

## Safety notes

- The default path makes zero OpenAI requests.
- Coding tools reject absolute paths, traversal, symlink escapes, secrets, evidence, and unrelated directories.
- Model-generated text is never passed to an unrestricted shell; only the allowlisted pytest command can run.
- The cost guard includes persisted ledger spending and the requested maximum output in its worst-case pre-request calculation.
- Do not publish `.env`, `runs/`, local database storage, private transcripts, or the usage ledger.
