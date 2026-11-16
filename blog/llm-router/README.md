# LLM routing with native VectorAI DB outcome memory

Seed a local/cloud model pair, retrieve checked historical outcomes with native vector search, choose the local tier when its evidence clears the gate, and store each attempt separately. Runtime validation and semantic feedback are distinct. Failures remain in the success-rate denominator; unverified answers never count as successes.

## Prerequisites

- Python 3.12 and the packages pinned in `requirements.lock.txt`.
- A running local [Actian VectorAI DB](https://docs.vectoraidb.actian.com/home/installation/instructions). The default gRPC address is `localhost:6674`; change `ACTIAN_VECTORAI_URL` if your server uses another local port. Setup creates a collection, not a server/container.
- [Ollama](https://ollama.com/) with the public [`qwen3:4b-instruct`](https://ollama.com/library/qwen3:4b-instruct) tag and `embeddinggemma:latest`. This example pins Qwen digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` and EmbeddingGemma digest `85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1`. If a public tag changes, the provider refuses the mismatched digest; do not silently substitute another model.
- An OpenAI API key and explicit permission to spend for paired cloud seeding/demo requests. The frozen cloud model is `gpt-5.4-2026-03-05`; its availability and recorded pricing assumptions must be checked before new paid use.
- Enough memory for the models. Requests are serial; models unload between embedding and generation. Guards require 1.5 GiB available before a request and stop after five sustained seconds below 512 MiB. This is not a guaranteed memory reservation.

## Install and configure

From this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
.\.venv\Scripts\python.exe -B -m llm_router.reader_tutorial config
.\.venv\Scripts\python.exe -B -m llm_router.reader_tutorial setup
```

Configure credentials privately in `.env`. Never commit it or overwrite an existing configuration blindly. Environment variables override the selected file; `--env-file .env.example` before the subcommand inspects credential-free defaults.

| Variable | Default / meaning |
|---|---|
| `ACTIAN_VECTORAI_URL` | `localhost:6674`, local plaintext gRPC |
| `ACTIAN_COLLECTION` | `vectorai_llm_router_tutorial_attempts_v1` |
| `ACTIAN_VECTORAI_API_KEY` | Blank; set privately if needed |
| `OLLAMA_HOST` | `http://localhost:11435` |
| `ROUTER_INFERENCE_PROFILE` | `qwen-memory-v2.json` |
| `LOCAL_MODEL` / `EMBEDDING_MODEL` | `qwen3:4b-instruct` / `embeddinggemma:latest` |
| `OLLAMA_NUM_CTX` | `2048`; effective generation settings are pinned by the profile |
| `OPENAI_MODEL` / `OPENAI_API_KEY` | Pinned cloud model / private key |
| `ALLOW_PAID_INFERENCE` | `false`; an environment value cannot enable paid execution |
| `EXPERIMENT_BUDGET_USD` | `0.75` cumulative ceiling |
| `EXPERIMENT_LEDGER` | `runs/experiment-budget.sqlite3`, shared across run IDs |

`setup` uses the [installed SDK's collection API](https://docs.vectoraidb.actian.com/sdks/python/reference) to create an unnamed 768-dimensional cosine collection when absent. It verifies dimension/metric when present and stops on incompatibility without resetting/deleting data. SDK 1.0.2 omits configuration from its parsed collection-info object, so compatibility verification reads the same SDK-generated Get RPC metadata. The connector supports local plaintext gRPC; remote TLS deployment needs separate configuration.

### Start Ollama and obtain the public models

In a separate terminal, if the project service is not already running:

```powershell
$env:OLLAMA_HOST='http://localhost:11435'
$env:OLLAMA_NUM_PARALLEL='1'
$env:OLLAMA_MAX_LOADED_MODELS='1'
$env:OLLAMA_CONTEXT_LENGTH='2048'
$env:OLLAMA_VULKAN='true'
$env:OLLAMA_NO_CLOUD='true'
ollama serve
```

In another terminal, download the public tags to that service:

```powershell
$env:OLLAMA_HOST='http://localhost:11435'
ollama pull qwen3:4b-instruct
ollama pull embeddinggemma:latest
```

Downloads are reader setup instructions, not commands used during publication verification. Existing services should not be stopped. The unchanged generation profile uses context 2048, output allowance 256, six threads, batch 32, seed 42 and GPU offload setting 99. Hardware/backend support and available memory remain external prerequisites.

## Fresh inference workflow

These commands run **new model inference**. Cloud calls require `--enable-paid`, a key, available budget and permission to spend. Without the flag, paired seeding/demo refuses before constructing providers. They were not executed against real models to validate this reader package.

```powershell
# Both-model seeding, one response per message/model.
.\.venv\Scripts\python.exe -B -m llm_router.reader_tutorial seed --models local,cloud --enable-paid
# Separate offline draft-reference check, then feedback upsert/readback.
.\.venv\Scripts\python.exe -B -m llm_router.reader_tutorial check --split seed
.\.venv\Scripts\python.exe -B -m llm_router.reader_tutorial apply-feedback
# Native model choice -> actual selected provider -> runtime check -> attempt write-back.
.\.venv\Scripts\python.exe -B -m llm_router.reader_tutorial demo --enable-paid
.\.venv\Scripts\python.exe -B -m llm_router.reader_tutorial check --split demo
.\.venv\Scripts\python.exe -B -m llm_router.reader_tutorial apply-feedback
# Native evidence inspection using this run's saved vector; no new inference.
.\.venv\Scripts\python.exe -B -m llm_router.reader_tutorial inspect --message-id D-004
```

The bundle contains 40 synthetic seeds and four demo messages in `tutorial_data/inputs/`. Draft references/rationales are separate in `tutorial_data/offline_labels/` and are read only by `check`, never routing or provider requests. They are AI-assisted synthetic labels pending independent human review, not verified ground truth. No evaluation labels are bundled.

This path generates its own state under `artifacts/reader-example-v1/`; it needs no private journals, historical run IDs or prior responses. For another run, put `--run-id your-run` before each subcommand. Completed seed responses/vectors are reused regardless of correctness; resume fills only missing model slots. Interrupted/error attempts stop for reconciliation, never automatic resend. Completed demo results are reused. Do not delete journals or change run IDs to evade these checks or the ledger. Local-only seeding is available with `seed --models local` without a paid flag; it does not authorize subsequent cloud fallback.

## Routing, checks and write-back

The [routing-loop diagram](ROUTING_LOOP.md) shows the separate native searches. `points.search` with `FilterBuilder.must` selects checked successes; another query retains failures for per-model, distinct-message rates. The existing exploratory gate is cosine minimum 0.5, five messages, at least 80% agreement, and up to ten neighbors. These are not validated reliability guarantees.

On the tested server, native filtered search needed full-collection over-fetch and `hnsw_ef >= 4096`. The adapter verifies every expected ID and fails closed on incomplete results. It refuses collections above 10,000 points. Scores/ranking come from VectorAI DB, not Python cosine ranking; this is a bounded workaround, not a scalability claim.

Runtime validation checks schema, clarification consistency and exact input order-ID membership, not semantic correctness. `check` separately compares saved seed/demo answers with draft references and emits bound boolean feedback/provenance. Raw attempts remain unchanged and semantically unverified without feedback. A local runtime rejection escalates once and preserves separate local/cloud UUIDs linked to the request. The provider-driven outcome loop also supports an explicit bound semantic checker; future live outcomes require reviewed feedback, not assumed cloud correctness. Four demo messages alone cannot satisfy five-neighbor support or prove learning.

The shared ledger reserves conservatively before requests, limits cloud output to 256 tokens, disables retries, and counts confirmed charges plus the full unresolved exposure against $0.75. It initializes only when absent; never replace/reset it to regain budget or create another ledger per run. A funded API account is not itself permission to spend. Pricing assumptions in code must be rechecked before new paid use. Local compute, memory and energy are not free or priced here.

## Verification and evidence limits

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m llm_router.reader_tutorial --env-file .env.example config
```

Deterministic tests use simulated providers/database responses. They verify parsing/configuration, paid-disabled refusal, idempotent setup/incompatibility, paired seeding/checking/feedback/routing/write-back without historical artifacts, native filter parameters, failure-inclusive rates, validation and persistent budget uncertainty. They make no model or paid calls.

In earlier real project execution, native retrieval and seed/demo storage/readback were verified, and collection setup was tested on a uniquely named disposable collection. **The fresh reader inference path has only fixture execution evidence.** No fresh model-backed run or new evaluation was performed for this publication package.

Earlier, separately preserved **saved-response replay** used 100 Qwen responses and 99 cloud responses; cloud E-001 remained unresolved. Qwen draft exact agreement was 58/100 (58/94 excluding schema-scope cases); cloud/replay agreement was 95/99 (89/93). Checkpoint 40 selected local twice and yielded $0.005786 / **2.01% counterfactual API savings** on the same 99 cases. These resumed diagnostic results came from a previously inspected synthetic evaluation set, not fresh held-out confirmation. Replay cost is not actual spending or live latency. [Recorded results and network-free verification](results/README.md) provide a minimal public export for auditing the counts, recorded choices, API costs and API-only seeding break-even. Reference correctness cannot be independently regraded from that export. No private journals are distributed and there is no saved-response model-execution command. The updated feedback loop has not received a new held-out evaluation.

`frozen/extraction-v1.1.0/` is byte-preserved because its manifest verifies all contract files, including historical documentation. Its old $3 proposal, early status notes and frozen dependency snapshot are provenance, not current instructions; the effective cap is $0.75 and installation uses the root lockfile. Do not edit those files without a new contract version. No general improvement, speed benefit, built-in database router, guaranteed model switch or zero local compute cost is claimed.
