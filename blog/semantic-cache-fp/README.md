# Semantic Cache False-Positive Tests

Code for the seven tests in *Stop Trusting Your Cache Hit Rate. 7 Tests Show Why*.

## Setup

Python 3.10 to 3.12 and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`, or `brew install uv`). uv downloads a matching Python if you don't have one.

```bash
docker-compose up -d                      # VectorAI DB on localhost:6574, Local UI on localhost:6575
uv sync                                   # creates .venv and installs everything in pyproject.toml
```

On an Apple Silicon Mac, if the image fails to start, add `platform: linux/amd64` under `vectorai:` in `docker-compose.yml`. On Linux or WSL2, if the container exits, run `sudo chown -R 999:999 ./local_data` and start it again.

## Run

Run test 1 first and pick the operating threshold from its output. Then pass that same `--threshold` to every other test so the numbers line up. The commands below use 0.80 as the example value.

```bash
uv run python test1_threshold_sweep.py --threshold 0.80
uv run python test2_negation.py        --threshold 0.80
uv run python test3_numbers_dates.py   --threshold 0.80
uv run python test4_partitions.py      --threshold 0.80
uv run python test6_query_length.py    --threshold 0.80
uv run python test7_stale_answers.py   --threshold 0.80 --change-frac 0.20
uv run python make_table.py
```

The first run downloads Banking77 and caches embeddings in `data/`. Each script takes a few minutes on a laptop.

### Test 5 needs a GPU

Colab and most GPU notebooks cannot run Docker, so test 5 runs in two steps:

```bash
# Step 1, on your laptop with VectorAI DB running
uv run python test5_decider_verify.py --threshold 0.80 --export-hits data/test5_lookups.json

# Step 2, on the GPU machine (upload the project folder and data/test5_lookups.json)
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pip install uv
uv sync --extra decider
uv run --extra decider python test5_decider_verify.py --threshold 0.80 --model Mapika/decider-0.8b --max-hits 100 --hits-file data/test5_lookups.json
```

Copy `results/test5.json` back to your laptop, then run `uv run python make_table.py`.

## Outputs

- `results/test1.json` to `results/test7.json`: every number, the full threshold sweep, and a `row` for the table.
- `results/test1_sweep.csv` and `results/test1_sweep.png`: the sweep data and curve.
- `data/pairs_negation.jsonl` and `data/pairs_numbers.jsonl`: every pair tests 2 and 3 used.
- `results/results_table.md`: the seven-row table.
