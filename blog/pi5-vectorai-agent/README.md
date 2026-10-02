# Local AI Agent on Raspberry Pi 5 with VectorAI DB

A fully offline AI agent with persistent semantic memory. It runs on a single Raspberry Pi 5 with no cloud API calls.

- **Ollama** runs `qwen2.5:1.5b` for chat and `nomic-embed-text` for embeddings
- **Actian VectorAI DB** stores the agent's memory in an HNSW index
- **agent.py** is the agent loop: under 100 lines of plain Python, no framework

## Hardware requirements

| Part | Requirement | Why |
|---|---|---|
| Board | Raspberry Pi 5, **8 GB** | 4 GB cannot hold VectorAI DB and Ollama together |
| Storage | NVMe SSD on a PCIe HAT for `data/` and `ollama/` (booting from SD is fine) | Loading models from an SD card is slow enough to cause timeouts |
| Cooling | Active cooler | Sustained inference throttles the CPU without it |
| OS | Raspberry Pi OS 64-bit (tested on Trixie) | The images are arm64 |
| Power | Official 27 W USB-C supply | Prevents undervoltage throttling under load |

Tested on: `<fill in: board revision, NVMe model, HAT, cooler, OS image date>`

No Pi? The same arm64 images run on Apple Silicon Macs. The commands are the same, but your benchmark numbers will be different.

## Quickstart

Clone onto the NVMe drive (for example under `/mnt/nvme`): `setup.sh` creates `data/` and `ollama/` next to the code, so the database and the models land on the fast drive.

```bash
git clone https://github.com/actian-devs/developer-content.git
cd developer-content/blog/pi5-vectorai-agent
chmod +x setup.sh && ./setup.sh
uv run agent.py
```

The setup script installs Docker and uv, starts both containers, pulls the models and runs smoke tests. `uv run` reads `pyproject.toml`, creates `.venv` and installs the dependencies on first use, so there is no separate install step.

## Memory budget

| Service | Cap | Notes |
|---|---|---|
| VectorAI DB | 2 GB | `mem_limit` in docker-compose.yml |
| Ollama | 4 GB | Qwen 2.5 1.5B (~1.1 GB) + nomic-embed-text (~0.3 GB) + KV cache |
| OS + agent | ~2 GB | |

To record your measured usage, run `docker stats --no-stream`.

Raspberry Pi OS boots with the memory cgroup disabled, so Docker ignores `mem_limit` (`docker info` warns `No memory limit support`). To enable the caps, add `cgroup_enable=memory` to the kernel command line and reboot:

```bash
sudo sed -i '1 s/$/ cgroup_enable=memory/' /boot/firmware/cmdline.txt && sudo reboot
```

After the reboot, recreate the containers once (`docker compose up -d --force-recreate`) so the limits apply. If you use a VectorAI DB licence, do this before activating it, because recreating the container invalidates the licence.

## Benchmark

```bash
uv run benchmark.py            # 1K / 10K / 100K vectors, 100 queries each
uv run plot_benchmark.py       # writes results/benchmark.png
```

The free Community edition caps VectorAI DB at 5,000 vectors. Without a licence, run `uv run benchmark.py --sizes 1000 2000 4000`.

The benchmark reports p50, p95 and p99 search latency for 768-dim cosine search with top-5 results. The script records the board, RAM, storage device holding `data/`, CPU temperature and throttle state in the CSV.

The vectors are synthetic, unit-normalized random data. Random vectors are harder for HNSW than real embeddings, so the numbers are conservative.

## Files

| File | Purpose |
|---|---|
| `docker-compose.yml` | VectorAI DB + Ollama with memory caps |
| `setup.sh` | From a fresh Pi to running services and smoke tests |
| `agent.py` | Agent loop: embed, retrieve, inject, infer, write back |
| `pyproject.toml` | Python dependencies, read by `uv run` |
| `benchmark.py` | Retrieval latency at 1K / 10K / 100K vectors |
| `plot_benchmark.py` | p50/p99 chart from the benchmark CSV |

## Links

- VectorAI DB docs: https://docs.vectoraidb.actian.com
- Docker install: https://docs.vectoraidb.actian.com/home/installation/instructions
- Python SDK: https://docs.vectoraidb.actian.com/sdks/python/installation
