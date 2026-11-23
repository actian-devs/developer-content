# Recording the Demo

A step-by-step guide for filming the video that goes with *Best Quantization for Phi-4, Gemma 4, and Qwen3 on a Pi 5*. The commands run in order on the Pi 5, each step shows the output to expect, and **Highlight** says what to put on screen. The expected output is trimmed from the run in `saved-results/`. Your numbers will differ a little, but the fit results and picks should match.

The video follows the article: the hook, the board and the models, what a level changes, the memory test, the quality test, then the picker.

## Before you film

- A Raspberry Pi 5 with 8 GB of RAM, the Active Cooler, the official 27 W power supply, Raspberry Pi OS 64-bit, and about 15 GB of free disk.
- Two terminal panes side by side, for example with `tmux`. Pane A runs the commands below. Pane B shows live memory.
- A terminal font large enough to read on a phone screen.
- Run steps 1 and 2 before the shoot. The build takes 15 to 20 minutes and the full test takes several hours, so film those as short cutaways rather than in real time.

In pane B, keep this running for the whole recording:

```bash
watch -n 1 free -h
```

**Highlight:** the `used` column on the `Mem:` row and the `used` column on the `Swap:` row. These are the two numbers the fit result comes from.

## 1. Clone the repo and set up the board

```bash
git clone https://github.com/actian-devs/developer-content.git
cd developer-content/blog/pi5-quantization-picker
./setup.sh
```

If the script installs Docker, log out, log back in, and run `./setup.sh` again.

Expected output at the end:

```text
Waiting for VectorAI DB...
...
Ready. Board:
Raspberry Pi 5 Model B Rev 1.1
               total        used        free      shared  buff/cache   available
Mem:           7.9Gi       ...
Swap:          2.0Gi       ...

Next: uv run run.py
```

**Highlight:** the board name and the `total` memory. Then show the two containers running:

```bash
docker ps --format "{{.Names}}  {{.Image}}"
```

```text
vectorai  actian/vectorai:1.0.3
ollama  ollama/ollama:0.40.15
```

## 2. Run one level that fits

Gemma 4 E4B at Q5_K_M is the largest level of that model that fits. Recording it gives you a fitting level for the memory segment and checks that the setup works.

```bash
uv run run.py --models gemma4-e4b --levels Q5_K_M
```

Expected output:

```text
Indexing 4000 catalog chunks in VectorAI DB...
Retrieval: the right chunk is in the top 3 for 16/16 answerable questions
Stack idle: board uses 1.296 GiB before any LLM loads
  Gemma 4 E4B Q5_K_M: fits, file 5.105 GiB, peak used 5.337 GiB, swap +0.115 GiB, quality 0.975
```

**Highlight:**
- `Stack idle: board uses 1.296 GiB`. This is the memory bar in Figure 1: VectorAI DB and the embedding model are loaded before any model starts.
- In pane B, `Mem: used` climbing as the model loads, while `Swap: used` barely moves.
- The result line: a 5.1 GiB file that fits.

## 3. Run the level that doesn't fit

This is the shot for the memory segment and the 60-second X clip.

```bash
uv run run.py --models gemma4-e4b --levels Q6_K
```

Expected output:

```text
  Gemma 4 E4B Q6_K: swaps, file 6.589 GiB, peak used 5.913 GiB, swap +1.561 GiB, quality 0.975
```

**Highlight:**
- In pane B, `Swap: used` climbing past 1.5 GiB while the model loads. Loading took about 106 seconds here, against about 20 seconds at Q5_K_M.
- `swap +1.561 GiB` and `swaps` on the result line.
- `quality 0.975`, the same score as Q5_K_M. The extra memory bought no quality.

## 4. Show VectorAI DB and the model answering together

This is the hook shot. Download the Gemma 4 E2B pick, serve it, and ask a catalog question end to end. Run the server in pane A, from the repo folder where `setup.sh` built llama.cpp:

```bash
wget -P models https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf
llama.cpp/build/bin/llama-server -m models/gemma-4-E2B-it-Q8_0.gguf -c 4096 -np 1 -t 4 -ngl 0 --jinja --reasoning off
```

Wait for these lines (each starts with a timestamp):

```text
srv  server_run_i: model loaded
srv  server_on_li: listening on http://127.0.0.1:8080
```

Once a question comes in, the server also logs `print_timing` lines with tokens per second. Keep pane A's log out of the shot or scrolled up while you film the answer.

In a third pane, from the same folder, ask the question:

```bash
uv run demo.py
```

Expected output (the catalog entries are the ones VectorAI DB returns on your board):

```text
Question: What is the maximum load of the Cobalt 41 clamp collar?

VectorAI DB top 3:
  - The Cobalt 41 clamp collar (part ...) carries a maximum load of 29.4 kg ...
  - ...
  - ...

Answer: The maximum load of the Cobalt 41 clamp collar is 29.4 kg.
```

**Highlight:** the retrieved entry that contains the answer, then the answer with the same 29.4 kg value. Keep pane B in frame so the memory is visible with both running. Open the video on this shot next to the picker table from step 7.

Stop the server with `Ctrl+C` when you are done.

## 5. Show one answer that changes with the level

For the quality segment, show the Phi-4 Mini JSON extraction from the saved run. It was correct at Q3_K_M and wrong at Q8_0.

```bash
grep -o 'name\\": \\"Halcyon 70[^\\]*' saved-results/phi4-mini/Q3_K_M.json
grep -o 'name\\": \\"Halcyon 70[^\\]*' saved-results/phi4-mini/Q8_0.json
```

Expected output:

```text
name\": \"Halcyon 70 gear housing
name\": \"Halcyon 70 gear housing (part NR-7B2-500)
```

**Highlight:** the `name` field. At Q8_0 the model added the part number `(part NR-7B2-500)`, so the field no longer matches the catalog.

## 6. The full run

Film this as a cutaway with a time-lapse or a few short shots, not in real time. It takes several hours. Run it inside `tmux` so it survives a dropped connection. If it stops, run the same command again and it picks up where it left off.

```bash
uv run run.py
```

Expected output, one line per level (24 in all). Levels that already have a result are skipped:

```text
  Phi-4 Mini Q3_K_M: fits, file 1.972 GiB, peak used 4.411 GiB, swap +0.052 GiB, quality 0.925
  ...
  Phi-4 Mini Q8_0: swaps, file 3.804 GiB, peak used 6.681 GiB, swap +0.71 GiB, quality 0.85
  ...
  Gemma 4 E4B Q5_K_M: done, skipping
  Gemma 4 E4B Q6_K: done, skipping
  ...
  Qwen3-4B Q8_0: swaps, file 3.986 GiB, peak used 6.884 GiB, swap +1.686 GiB, quality 0.925
```

**Highlight:** Phi-4 Mini Q8_0 swapping with a 3.8 GiB file, against the 5.1 GiB Gemma 4 E4B Q5_K_M that fit in step 2. File size doesn't predict fit.

Then the vector-side test. It rebuilds the catalog under each index type, which takes a few minutes:

```bash
uv run vector_side.py
```

Expected output:

```text
{'index': 'hnsw', 'vectors': 4000, ..., 'container_growth_mib': 86.2, 'recall_at_10': 1.0, 'rag_top3_hits': '16/16'}
{'index': 'ivf_flat', 'vectors': 4000, ..., 'container_growth_mib': 83.7, 'recall_at_10': 0.988, 'rag_top3_hits': '14/16'}
{'index': 'ivf_pq', 'vectors': 4000, ..., 'container_growth_mib': 67.4, 'recall_at_10': 0.835, 'rag_top3_hits': '13/16'}
Wrote results/vector_side.json; catalog rebuilt. Raw float32 vectors: 11.7 MiB (0.011 GiB)
```

**Highlight:** `container_growth_mib` and `recall_at_10` on the `ivf_pq` line. Product quantization freed about 19 MiB and cost recall on a catalog this size.

## 7. The picker

```bash
uv run report.py
```

This prints every table and writes `results/tables.md` and `results/summary.csv`. The last table is the picker:

```text
## Picker

Pick = the highest level that matches the model's best score and grows swap by no more than 0.15 GiB (0.10 GiB under the fit limit). ...

| Model | Pick | Fallback | Score at pick |
| --- | --- | --- | --- |
| Phi-4 Mini | Q4_0 | Q3_K_M | 92% |
| Gemma 4 E2B | Q8_0 | Q5_K_M | 98% |
| Gemma 4 E4B | Q5_K_M | Q4_K_M | 98% |
| Qwen3-4B | Q5_K_M | Q4_0 | 92% |
```

**Highlight:** fill the picker table in one row at a time. This is the closing shot and the 90-second LinkedIn clip.

## Clips

| Clip | Length | Source |
| --- | --- | --- |
| X | 60 s | Step 3, Gemma 4 E4B Q6_K pushing the board into swap |
| LinkedIn | 90 s | Step 7, the picker table filling in |
| YouTube Short | under 60 s | Steps 2 and 3, `Mem: used` and `Swap: used` in pane B for a fitting level and a swapping one |

## What not to say on camera

- No tokens-per-second figures, and no comparisons with other sites' Pi speeds.
- No VectorAI DB speed or latency claims.
- Keep `llama-server`'s `print_timing` lines, which show tokens per second, off screen.
- The board throttled at some point during the saved run, so quote memory and quality, not timings. The load times in step 3 are the one exception, and only to show the cost of swap.
