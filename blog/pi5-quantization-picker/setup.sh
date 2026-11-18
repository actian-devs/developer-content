#!/bin/bash
# setup.sh: a fresh Raspberry Pi 5 to a ready test bench
# OS: Raspberry Pi OS 64-bit (Debian 13 trixie). Run from the repo folder on the NVMe drive:
# the models (up to 8 GB each) and the vector data are written here.
# Safe to run again: every step skips what is already done.
set -e   # stop at the first command that fails

LLAMA_CPP_TAG=v0.9.0   # the llama.cpp release every test runs on

# ── 1. System packages: git, the C/C++ compiler, CMake, and the SSL headers llama.cpp's server links against ──
sudo apt-get update
sudo apt-get install -y git cmake build-essential curl libcurl4-openssl-dev libssl-dev

# ── 2. Docker, which runs VectorAI DB and Ollama ──
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"   # lets this user run docker without sudo, after logging back in
    echo "Docker installed. Log out and back in (or run: newgrp docker), then run ./setup.sh again."
    exit 0
fi

# ── 3. uv, the Python package manager, and the project's Python environment ──
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync   # installs the exact package versions pinned in uv.lock

# ── 4. llama.cpp, built for the Pi 5's Cortex-A76 cores (GGML_NATIVE picks up NEON and dotprod) ──
if [ ! -x llama.cpp/build/bin/llama-server ]; then
    # clone only once; a rerun after a failed build reuses the folder instead of failing on it
    if [ ! -d llama.cpp ]; then
        git clone --depth 1 --branch "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp.git
    fi
    cmake -S llama.cpp -B llama.cpp/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
    # build only llama-server, the program that loads a model and answers over HTTP (about 15 to 20 minutes)
    cmake --build llama.cpp/build --config Release -j 4 --target llama-server
fi

# ── 5. Start VectorAI DB and Ollama ──
# VectorAI runs as uid 999; a root-owned bind mount crash-loops it
mkdir -p data ollama models results
sudo chown 999:999 data
docker compose up -d

echo "Waiting for VectorAI DB..."
until curl -sf http://localhost:6573/ > /dev/null 2>&1; do sleep 3; done

# ── 6. The embedding model (about 274 MB) ──
docker exec ollama ollama pull nomic-embed-text

# ── 7. Show the board, its memory and the free disk space ──
echo
echo "Ready. Board:"
tr -d '\0' < /proc/device-tree/model; echo
free -h
df -h .
echo
echo "Next: uv run run.py"
