# Which Quantization Level Fits a Raspberry Pi 5?

Test harness for *Best Quantization for Phi-4, Gemma 4, and Qwen3 on a Pi 5*. It measures, on one Raspberry Pi 5, which GGUF quantization level of each model fits in memory next to a running retrieval stack, and how well each level answers a fixed set of prompts.

- **Models:** Phi-4 Mini (3.8B), Gemma 4 E2B, Gemma 4 E4B and Qwen3-4B (thinking off), all from Unsloth's GGUF repos on Hugging Face
- **Levels:** Q3_K_M, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0
- **Runtime:** llama.cpp `llama-server`, built on the Pi, CPU only, 4 threads, 4,096-token context
- **Retrieval stack, running during every test:** VectorAI DB 1.0.3 holding a 4,000-chunk catalog in an HNSW index, and Ollama serving `nomic-embed-text`

## What it measures

For each model and level, `run.py`:

1. downloads the GGUF file,
2. starts `llama-server` with the embedding model and VectorAI DB already loaded,
3. samples memory every 0.25 seconds while the model loads and answers: whole-board RAM in use (MemTotal minus MemAvailable), swap growth, the `llama-server` process, and each container,
4. runs 40 prompts and scores each answer with an automatic check,
5. stops the server and deletes the file (keep it with `--keep`).

A level **fits** when it loads, answers all 40 prompts, and grows swap by no more than 0.25 GiB. It is **swaps** when it finishes but pushes the board into swap, **swap-bound** when prompts time out (180 seconds each, three strikes), and **fails to load** when the process dies or never becomes ready.

### The prompt set

The answers are checked by code, not by a judge model, so a score is reproducible.

| Kind | Prompts | Check |
| --- | --- | --- |
| RAG | 20 | Answer from the top 3 chunks VectorAI DB returns. 12 single-fact questions, 4 two-fact questions, 4 about products not in the catalog (the right answer says so). The answer must contain the record's exact values. |
| JSON | 10 | Extract a catalog entry into JSON with five named keys and types. Must parse, have exactly those keys, and match the record. |
| Instructions | 10 | Short, checkable instructions: exactly three bullets, at most 15 words, lowercase only, a reversed word in capitals, and so on. |

The catalog is fictional and generated from a fixed seed (`qpick/corpus.py`), so no model can answer from memory, and every run builds the same one. Retrieval runs once and is cached, so every model and level answers from exactly the same chunks.

## Run it on the Pi

Use a Pi 5 with active cooling, and run from the NVMe drive: the files are 2 to 8 GB each, and the run downloads about 90 GB in total (one file at a time, deleted after each test).

```bash
git clone https://github.com/actian-devs/developer-content.git
cd developer-content/blog/pi5-quantization-picker
chmod +x setup.sh
./setup.sh
```

`setup.sh` installs Docker and uv, builds llama.cpp (tag `v0.9.0`) for the Pi's Cortex-A76 cores, starts VectorAI DB and Ollama, and pulls `nomic-embed-text`. If it installs Docker, log out and back in, then run it again.

Then run the whole matrix (several hours; it resumes where it stopped if interrupted):

```bash
uv run run.py
```

One model or a few levels at a time:

```bash
uv run run.py --models gemma4-e4b --levels Q4_K_M Q5_K_M
```

Then the vector-side test and the tables:

```bash
uv run vector_side.py
uv run report.py
```

`vector_side.py` builds the same catalog under HNSW, IVF, and IVF with product quantization, and reports VectorAI DB's index memory and the recall cost. Add `--scalar` to include IVF with scalar quantization.

## Output

- `results/<model>/<level>.json`: memory peaks, fit result, every prompt with its answer and pass or fail
- `results/<model>/<level>.server.log`: the `llama-server` log for that level
- `results/cache/idle.json`: board memory with only the retrieval stack running, plus the board, OS, swap setup, storage, and llama.cpp, VectorAI DB and Ollama versions
- each level's JSON also records the Pi's temperature and throttle flags before and after the test
- `results/vector_side.json`: index memory and recall for each VectorAI DB index type
- `results/tables.md` and `results/summary.csv`: the fit, quality and picker tables

## Notes

- **Community Edition limit.** VectorAI DB Community Edition caps the server at 5,000 vectors, which is why the catalog has 4,000 chunks. That is a licence limit, not a limit of the board.
- **Swap.** The fit result depends on the board's swap setup; `results/cache/idle.json` records the swap size the run saw.
- **Speed.** The results record how long each answer took so that swap-bound levels show up, but the harness is not a speed benchmark.
