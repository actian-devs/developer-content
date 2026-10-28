# Entity Resolution With VectorAI DB and Gemini 4 Argon

Resolves duplicate records on public benchmarks two ways and measures both:

1. **Best-match threshold (baseline):** VectorAI DB blocking, then each record's nearest candidate is a match if it clears a similarity cutoff. (A naive global cutoff over all candidates is also reported, for reference.)
2. **Threshold + Argon:** the same blocking; nearest candidates above an upper edge auto-match, everything below a lower edge is rejected, and Gemini 4 Argon decides only the band: nearest candidates between the edges plus near-ties (2nd and 3rd candidates scoring almost as high as the best). Each verdict comes with a reason.

It reports the reduction ratio and pair completeness of blocking; precision, recall, and F1 for both approaches; the share of pairs sent to Argon; token cost; and timings.

## How it works

| Stage | What happens | Where |
|---|---|---|
| Load | Downloads the benchmark (two tables + gold matches) from the Leipzig collection | `er/data.py` |
| Embed | Flattens each record's fields to text and embeds it locally (Ollama by default) | `er/embed.py` |
| Block | Upserts every record into VectorAI DB; each left record's K nearest right records become candidates. A `must` filter restricts the search to the other table | `er/block.py` |
| Bands | Splits left records into dev/test. On dev: the best-match cutoff (baseline) and the band edges. The band adds near-tie candidates. Everything is reported on test | `er/bands.py` |
| Adjudicate | Sends only the band's pairs to Argon with structured output; caches every verdict and reason in `argon_verdicts.jsonl` | `er/adjudicate.py` |

Recall counts true matches lost at blocking as misses, so pair completeness caps it for both approaches.

## Datasets

| Dataset | Records | True matches |
|---|---|---|
| `abt-buy` | 1,081 Abt + 1,092 Buy = 2,173 | 1,097 |
| `amazon-google` | 1,363 Amazon + 3,226 Google = 4,589 | 1,300 |

Both fit under the 5,000-vector cap of VectorAI DB Community Edition. Source: [Leipzig DB group benchmark collection](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).

## Quickstart

Requires Docker, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) for embeddings, and a key for Gemini 4 Argon (it has no free tier).

```bash
docker compose up -d                          # VectorAI DB on localhost:6574
ollama pull embeddinggemma                    # local embedding model on localhost:11434
uv run resolve.py --dataset abt-buy --no-argon   # blocking + threshold only, no API calls
export OPENROUTER_API_KEY="your-key"          # PowerShell: $env:OPENROUTER_API_KEY="your-key"
uv run resolve.py --dataset abt-buy --argon-backend openrouter --workers 2 --budget-usd 10
```

Switch datasets with `--dataset amazon-google`. Only one dataset fits under the Community edition's 5,000-vector cap at a time; the harness drops its own earlier collections, and if the server still reports the cap, run `docker compose restart vectorai`.

To embed in-process with a Hugging Face model instead of Ollama: `uv sync --extra st`, then `--embed-backend sentence-transformers --embed-model BAAI/bge-small-en-v1.5`.

## Environment variables

Set only the one for the Argon backend you use. Keys are read from the environment and never written to disk.

| Variable | Backend | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `--argon-backend openrouter` | Prepaid credit; OpenRouter routes to Google's endpoints and reports the exact cost of each call. Keep `--workers` low (2) to stay under its rate limit. |
| `GEMINI_API_KEY` | `--argon-backend ai-studio` (default) | Google AI Studio key on a project with billing turned on. |
| `GOOGLE_CLOUD_API_KEY` | `--argon-backend agent-platform` | Google Cloud (Gemini Enterprise Agent Platform). Alternatively use Application Default Credentials with `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` (default `global`). |

## Measured results

From the runs behind the article (laptop: Intel Core i7-6600U, 8 GB RAM, CPU only; VectorAI DB 1.0.2; EmbeddingGemma; Gemini 4 Argon at thinking level LOW through OpenRouter):

| Dataset | Reduction ratio | Pair completeness | F1, best-match threshold | F1, with Argon | Share sent to Argon | Argon cost |
|---|---|---|---|---|---|---|
| Abt-Buy | 99.08% | 99.0% | 0.813 | 0.915 | 13.2% | $1.41 |
| Amazon-Google | 99.69% | 94.1% | 0.597 | 0.691 | 14.0% | $2.42 |

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `abt-buy` | `abt-buy` or `amazon-google` |
| `--k` | 10 | neighbours per record kept as candidates |
| `--sweep-k` | 1 3 5 10 20 | K values reported in the blocking table |
| `--hnsw-ef` | 256 | HNSW search breadth (the filtered search also over-fetches 10 x K and retries any record still short) |
| `--dev-share` | 0.3 | share of left records used to tune the cutoff and band edges |
| `--target-precision` | 0.95 | precision required above the upper edge |
| `--max-miss` | 0.05 | share of candidate matches allowed below the lower edge |
| `--near-tie-rank` | 3 | lowest-ranked candidate that can enter the band as a near-tie |
| `--near-tie-gap` | 0.03 | a rank 2..R candidate enters the band if it scores within this of its record's best |
| `--band-scale` | 1.0 | widen or narrow the band around its centre (2 doubles it) |
| `--embed-backend` | `ollama` | `ollama` or `sentence-transformers` |
| `--embed-model` | `embeddinggemma` | embedding model name for the chosen backend |
| `--ollama-url` | `http://localhost:11434` | Ollama server for embeddings |
| `--argon-model` | `gemini-4-argon` | Gemini model ID |
| `--argon-backend` | `ai-studio` | `ai-studio` (GEMINI_API_KEY), `agent-platform` (Google Cloud: GOOGLE_CLOUD_API_KEY, or ADC with GOOGLE_CLOUD_PROJECT), or `openrouter` (prepaid credit, OPENROUTER_API_KEY; same Google-served model, exact per-call cost) |
| `--thinking-level` | `LOW` | Argon's thinking level (`LOW`, `MEDIUM`, `HIGH`, `default`); thinking tokens are billed |
| `--workers` | 8 | concurrent Argon requests (use 2 with OpenRouter) |
| `--budget-usd` | 10.0 | hard cap on total Argon spend across all runs, counted from the verdict logs; a run that hits it stops and reports no partial accuracy |
| `--max-argon-pairs` | 3000 | stop before calling Argon if the band is larger than this |
| `--price-in` / `--price-out` | 2.00 / 10.00 | USD per 1M input / output tokens (Argon pricing through 30 Nov 2026) |

## Output

Written to `results/<dataset>/`:

- `summary.json`: every number above, including the blocking sweep
- `candidates.csv`: each candidate pair with its score, split, gold label, zone, and Argon's verdict and reason
- `argon_verdicts.jsonl`: one line per Argon call with tokens and latency; reruns reuse it and never pay twice
- `score_distribution.png`: true matches against non-matches, with the band and the best cutoff marked

Cost per 1,000 records is computed over the test split: the test left records plus the full right table.

## Saved results

`saved-results/<dataset>/` holds the output of the runs behind the article, in the same format, so you can compare your own run with ours: the blocking sweep and scores in `summary.json`, every candidate pair with its zone and Argon's verdict in `candidates.csv`, and every Argon call with its reason, tokens and exact OpenRouter charge in `argon_verdicts.jsonl`. The verdict logs also hold the calls from the band-width runs (`--band-scale 0.5`, `1.5` and `2`).

They sit outside `results/` on purpose: the harness reuses any verdicts it finds in `results/`, so copying ours there would replay them instead of calling Argon, and would count our spend against your `--budget-usd`.
