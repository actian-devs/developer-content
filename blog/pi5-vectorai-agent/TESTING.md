# Testing on the Pi: checklist and shot list

Use this to confirm the stack works and to capture every number and screenshot the article needs.
Each step lists what "good" looks like. Examples assume the Pi is `ophelia@192.168.1.140` and the project is in `~/pi5_local_agent`.

## 0. Copy the latest scripts to the Pi

The version on the Pi prints no latency line and no p95 or temperature data. Back it up, then copy the repo versions over. Run this from Windows PowerShell in the repo folder:

```powershell
ssh ophelia@192.168.1.140 "mkdir -p ~/pi5_agent_backup && cp ~/pi5_local_agent/*.py ~/pi5_agent_backup/"
scp agent.py benchmark.py plot_benchmark.py requirements.txt README.md TESTING.md docker-compose.yml ophelia@192.168.1.140:~/pi5_local_agent/
```

On the Pi, install the dependencies into the venv: `cd ~/pi5_local_agent && uv pip install -r requirements.txt`.

## 1. Health checks

SSH in (`ssh ophelia@192.168.1.140`), `cd ~/pi5_local_agent && source .venv/bin/activate`, then:

| Check | Command | Good result |
|---|---|---|
| Board | `cat /proc/device-tree/model; echo` | `Raspberry Pi 5 Model B Rev 1.x` |
| Storage | `findmnt -no SOURCE -T data; findmnt -no SOURCE -T ollama` | both on `/dev/nvme0n1p1` (the Pi boots from SD; only `data/` and `ollama/` live on the NVMe) |
| NVMe link | `sudo lspci -vv \| grep LnkSta:` | `Speed 8GT/s, Width x1` (PCIe Gen 3) |
| Licence | `curl -s localhost:6573/licenses/status` | `"state": "licensed"`, `"max_vectors": -1` |
| Containers up | `docker ps --format "{{.Names}}: {{.Status}}"` | `vectorai` and `ollama` both `Up` |
| Models pulled | `docker exec ollama ollama list` | `qwen2.5:1.5b` and `nomic-embed-text` |
| VectorAI DB | `python -c 'from actian_vectorai import VectorAIClient; print(VectorAIClient("localhost:6574").health_check())'` | a dict with `title` and `version` |
| Throttling | `vcgencmd get_throttled` | `throttled=0x0` |
| Temperature | `vcgencmd measure_temp` | under 60°C at idle |

Copy the **board, storage device, VectorAI DB version and kernel (`uname -r`)** into the article's hardware placeholders.

## 2. Agent demo (the opening screenshot)

The agent now prints a timing line after each answer. That line is the screenshot. The demo only works if you give it something to remember, so run two sessions.

**Session 1: teach it something.**

```bash
python agent.py
```

Expected banner:

```text
Agent ready on Raspberry Pi 5 Model B Rev 1.x, 8 GB RAM | LLM: qwen2.5:1.5b | 0 memories in VectorAI DB. Ctrl+C to quit.
```

Type these one at a time:

1. `What hardware are you running on?`: the answer should name the Pi 5 and 8 GB, because the agent now knows its own hardware.
2. `Remember this: greenhouse sensor 3 read 41 degrees at 2pm today, the other sensors read 28.`
3. `Also note: the irrigation pump in zone B was switched off for maintenance.`

Press **Ctrl+C** to quit. Ctrl+V does nothing here.

**Session 2: prove memory persists.** Optionally run `docker compose restart vectorai` first to make the point stronger. Then:

```bash
python agent.py
```

The banner should now show `3 memories`. Ask:

```text
you> Which greenhouse sensor was reading high, and what was going on in zone B?
```

Expected format (illustrative numbers):

```text
  [embed 110 ms | retrieval 3.4 ms (2 memories) | inference 9.8 s | write-back 130 ms]
agent> Sensor 3 was reading high at 41 degrees ... the zone B irrigation pump was off for maintenance.
```

(Your numbers will differ.) **Screenshot this: it's the article's opening shot and the video's hook.** Copy the numbers into `[[EMBED_MS]]`, `[[RETRIEVAL_MS]]`, `[[INFERENCE_S]]`, `[[WRITEBACK_MS]]`.

If it says `(0 memories)`, the similarity threshold is filtering them out. Lower `MIN_SCORE` in `agent.py` from `0.5` to `0.35` and ask again.

Tokens per second for `[[TOKENS_PER_SEC]]`:

```bash
curl -s localhost:11434/api/generate -d '{"model":"qwen2.5:1.5b","prompt":"Explain HNSW in 3 sentences.","stream":false}' | python3 -c 'import json,sys; r=json.load(sys.stdin); print(round(r["eval_count"]/r["eval_duration"]*1e9,1), "tok/s")'
```

## 3. RAM split

While the agent is running (both models loaded), in a second SSH window:

```bash
docker stats --no-stream --format "{{.Name}}: {{.MemUsage}}"
free -h
```

Screenshot it. It fills `[[OLLAMA_RAM]]`, `[[VECTORAI_RAM]]`, `[[OS_IDLE_RAM]]`, `[[TOTAL_RAM]]`.

## 4. Benchmark

The default sizes are 1K / 10K / 100K, which needs a licence. The free Community edition caps the server at 5,000 vectors; without a licence, run `python benchmark.py --sizes 1000 2000 4000`.

```bash
python benchmark.py
python plot_benchmark.py
```

Good result: a p50/p95/p99 line per size, then `Wrote results/benchmark.csv. End state: temp=XX.X'C, throttled=0x0`. `plot_benchmark.py` writes `results/benchmark.png`, which is the chart and the social asset. Copy it back to Windows:

```powershell
scp ophelia@192.168.1.140:~/pi5_local_agent/results/benchmark.* .
```

**Adding a licence:** no web UI is needed. On the Pi, this prompts for the product key, so it stays out of your shell history:

```bash
read -rsp "VectorAI product key: " KEY; echo; curl -s -X POST localhost:6573/licenses/add -H "Content-Type: application/json" -d "{\"product_key\":\"$KEY\"}"; unset KEY
```

Within 30 s, `curl -s localhost:6573/licenses/status` should show `"state": "licensed"`.

## 5. Thermal check (optional, for the hardware section)

Run the benchmark twice back to back with `watch -n 2 vcgencmd measure_temp` in a second window. Note the peak temperature and whether `get_throttled` is still `0x0`. If you can, repeat once without the cooler for the comparison sentence.

## What to send back for the article

- [ ] Session 2 screenshot with the timing line
- [ ] Answer to "What hardware are you running on?"
- [ ] `docker stats` + `free -h` output
- [ ] `results/benchmark.csv` and `results/benchmark.png`
- [ ] Board model, NVMe drive, HAT, cooler, kernel version
- [ ] Tokens per second
- [ ] Peak temperature and throttle value
- [ ] Licence status and expiry date (`curl -s localhost:6573/licenses/status`)
