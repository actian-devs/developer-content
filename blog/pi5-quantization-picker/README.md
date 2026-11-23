# Which Quantization Level Fits a Raspberry Pi 5?

This repo measures which GGUF quantization level of four small language models fits on a Raspberry Pi 5 with 8 GB of RAM, while a retrieval stack (VectorAI DB and an embedding model) runs on the same board, and how well each level answers a fixed set of prompts. It is the test harness behind the article *Best Quantization for Phi-4, Gemma 4, and Qwen3 on a Pi 5*, and `saved-results/` holds the run the article reports.

## The short answer

On an 8 GB Pi 5 with VectorAI DB and `nomic-embed-text` running:

| Model | Pick | Fallback | Why |
| --- | --- | --- | --- |
| Phi-4 Mini | Q4_0 | Q3_K_M | both scored 37 of 40, the model's best; Q4_K_M to Q6_K scored 35, and Q6_K sits at the swap limit |
| Gemma 4 E2B | Q8_0 | Q5_K_M | every level fits with almost no swap; both scored 39 of 40 |
| Gemma 4 E4B | Q5_K_M | Q4_K_M | both scored 39 of 40; Q6_K and Q8_0 push the board more than 1.5 GiB into swap without scoring higher |
| Qwen3-4B | Q5_K_M | Q4_0 | five levels tied at 37 of 40; Q6_K sits 0.03 GiB under the swap limit and Q8_0 grows swap by 1.69 GiB |

The rule: pick the highest level that matches the model's best score and stays at least 0.1 GiB under the 0.25 GiB swap limit, so one more process on the board doesn't tip it into swap.

Three findings from the run:

- **File size doesn't predict fit.** A 3.8 GiB Phi-4 Mini Q8_0 file pushed the board into swap, while a 5.1 GiB Gemma 4 E4B Q5_K_M file fit.
- **Quality barely changes with the level.** Every level of every model answered all 20 retrieval questions correctly, and overall scores stayed between 34 and 39 of 40.
- **The retrieval stack is small.** Before any model loads, VectorAI DB uses 0.15 GiB and the embedding model 0.26 GiB, leaving 6.57 GiB for the LLM.

The full tables are in [`saved-results/tables.md`](saved-results/tables.md).

## What the harness does

For each of 24 combinations (4 models x 6 levels), `run.py`:

1. downloads the GGUF file from Hugging Face,
2. starts llama.cpp's `llama-server` with VectorAI DB and the embedding model already loaded,
3. samples memory every 0.25 seconds while the model loads and answers: whole-board RAM in use, swap, the `llama-server` process, and each Docker container,
4. sends 40 prompts and scores every answer with an automatic check,
5. saves the result and deletes the model file, so only one file is on disk at a time.

**Models:** Phi-4 Mini (3.8B), Gemma 4 E2B, Gemma 4 E4B and Qwen3-4B, from Unsloth's GGUF repositories. Unsloth doesn't publish Phi-4 Mini at Q4_0, so that one file comes from bartowski's repository.

**Levels:** Q3_K_M, Q4_0, Q4_K_M, Q5_K_M, Q6_K and Q8_0.

**Server settings, the same for every level:** llama.cpp v0.9.0 built on the Pi, CPU only, 4 threads, one request slot, a 4,096-token context, and thinking mode off.

### When a level fits

| Result | Meaning |
| --- | --- |
| fits | loads, answers all 40 prompts, and grows swap by no more than 0.25 GiB |
| swaps | answers all 40 prompts but grows swap by more than 0.25 GiB |
| swap-bound | answers time out (180 seconds each, three strikes) |
| fails to load | the server dies or never becomes ready |

Peak RAM is MemTotal minus MemAvailable, the memory Linux could not give back. llama.cpp maps the model file into memory, and file pages Linux can reload count as available, so a model can peak below its own file size.

### The prompt set

Answers are checked by code, not by a judge model, so every score can be reproduced.

| Kind | Prompts | Check |
| --- | --- | --- |
| RAG | 20 | VectorAI DB returns the top 3 entries from a 4,000-entry parts catalog. 16 questions have an answer in the catalog and must contain its exact values. 4 ask about products that aren't there, and the answer must say so. |
| JSON | 10 | Extract a catalog entry into JSON with five named keys. It must parse, have exactly those keys and types, and match the entry. |
| Instructions | 10 | Short, checkable instructions, such as exactly three bullet points, at most 15 words, or lowercase only. |

The catalog is fictional and generated from a fixed seed, so no model can answer from what it learned in training and every run builds the same one. Retrieval runs once, so every model and level answers from the same retrieved entries.

### The vector side

`vector_side.py` builds the same catalog as an HNSW index, an IVF index, and an IVF index with product quantization, and records how much memory each adds to the VectorAI DB container and its recall against an exact search. `--scalar` adds IVF with scalar quantization.

## Run it on a Pi 5

You need a Raspberry Pi 5 (8 GB) with active cooling, Raspberry Pi OS 64-bit, and about 15 GB of free disk. No API keys or accounts.

```bash
git clone https://github.com/actian-devs/developer-content.git
cd developer-content/blog/pi5-quantization-picker
chmod +x setup.sh
./setup.sh
```

`setup.sh` installs Docker and uv, builds llama.cpp v0.9.0 for the Pi's Cortex-A76 cores (about 15 to 20 minutes), starts VectorAI DB 1.0.3 and Ollama, and pulls `nomic-embed-text`. If it installs Docker, log out and back in and run it again. If your shell doesn't find `uv` afterwards, run `source $HOME/.local/bin/env`.

Try one level first (about 15 minutes):

```bash
uv run run.py --models qwen3-4b --levels Q4_K_M
```

Then the full run, which takes several hours. Run it inside `tmux` so it survives a dropped connection; if it stops, run the same command again and it continues where it left off:

```bash
uv run run.py
```

Then the vector-side test and the tables:

```bash
uv run vector_side.py
uv run report.py
```

Options: `--models` and `--levels` pick a subset, `--redo` reruns levels that already have results, and `--keep` keeps each model file after its test.

## Files

| Path | What it does |
| --- | --- |
| `setup.sh` | Prepares a fresh Pi: Docker, uv, llama.cpp, VectorAI DB, Ollama, the embedding model |
| `docker-compose.yml` | VectorAI DB 1.0.3 and Ollama 0.40.15, without memory caps so the readings are real |
| `run.py` | The main test: every model at every level, from download to saved result |
| `vector_side.py` | Index memory and recall for HNSW, IVF and IVF with product quantization |
| `demo.py` | Asks one catalog question end to end, VectorAI DB retrieving and `llama-server` answering, for the video |
| `report.py` | Builds the fit, quality and picker tables from the result files, using the picker rule above |
| `qpick/models.py` | The models, levels and Hugging Face file for each |
| `qpick/corpus.py` | The fictional catalog and the 20 RAG questions |
| `qpick/prompts.py` | The 40 prompts and the check for each |
| `qpick/stack.py` | Embeds the catalog with Ollama, loads it into VectorAI DB, retrieves for each question |
| `qpick/server.py` | Starts `llama-server` for one file, sends prompts, stops it |
| `qpick/memory.py` | The background sampler that records peak memory |
| `saved-results/` | The run behind the article (see below) |
| `RECORDING.md` | Step-by-step guide for recording the demo video |

## Output

A run writes to `results/`:

- `results/<model>/<level>.json`: the board, file size, memory peaks, fit result, Pi temperature before and after, and every prompt with its answer and pass or fail
- `results/cache/idle.json`: memory with only the retrieval stack running, the board, the swap setup, and the VectorAI DB and Ollama versions
- `results/vector_side.json`: the vector-side results
- `results/tables.md` and `results/summary.csv`: the tables, from `report.py`

## Saved results

`saved-results/` holds the run the article reports, in the same format: every per-level result with all 40 answers, the idle stack reading, the retrieved entries, the vector-side results, and the tables. Compare your own run with it. It sits outside `results/` so your run never mixes with it.

That run used a Raspberry Pi 5 Model B with 8 GB of RAM, booting and storing everything on a microSD card, with 2 GiB of zram swap, the official Active Cooler and 27 W power supply, on Raspberry Pi OS (Debian 13, kernel 6.18).

## Notes

- **Licence key.** To run VectorAI DB with your own licence key, such as one from the [30-day full-featured free trial](https://www.actian.com/databases/vectorai-db/), follow the [install guide](https://docs.vectoraidb.actian.com/home/installation/instructions).
- **Swap.** Fit results depend on the board's swap setup; `idle.json` records the one each run saw.
- **One run.** With 40 prompts, one answer is 2.5 points, so a gap of one or two answers between levels is within what a single run can show.
- **Not a speed benchmark.** Each answer's time is recorded only so that swap-bound levels show up. The test board's throttle flags (`thermals_before` and `thermals_after` in each result) show it throttled at some point during the run, which affects speed but not memory use or answers.
- **Context length.** Every level ran with a 4,096-token context. A longer context needs more memory, so the fit results only hold at 4,096.
