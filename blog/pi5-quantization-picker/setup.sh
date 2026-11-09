#!/bin/bash
# setup.sh: a fresh Raspberry Pi 5 to a ready test bench
# OS: Raspberry Pi OS 64-bit (Debian 13 trixie). Run from the repo folder on the NVMe drive:
# the models (up to 8 GB each) and the vector data are written here.
set -e

LLAMA_CPP_TAG=v0.9.0

sudo apt-get update
sudo apt-get install -y git cmake build-essential curl libcurl4-openssl-dev

if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "Docker installed. Log out and back in (or run: newgrp docker), then run ./setup.sh again."
    exit 0
fi

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

# llama.cpp, built for the Pi 5's Cortex-A76 cores (GGML_NATIVE picks up NEON and dotprod)
if [ ! -x llama.cpp/build/bin/llama-server ]; then
    git clone --depth 1 --branch "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp.git
    cmake -S llama.cpp -B llama.cpp/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build llama.cpp/build --config Release -j 4 --target llama-server
fi

# VectorAI DB runs as uid 999; a root-owned bind mount crash-loops it
mkdir -p data ollama models results
sudo chown 999:999 data
docker compose up -d

echo "Waiting for VectorAI DB..."
until curl -sf http://localhost:6573/ > /dev/null 2>&1; do sleep 3; done
docker exec ollama ollama pull nomic-embed-text

echo
echo "Ready. Board:"
tr -d '\0' < /proc/device-tree/model; echo
free -h
df -h .
echo
echo "Next: uv run run.py"
