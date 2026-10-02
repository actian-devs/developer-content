#!/bin/bash
# setup.sh — Fresh Raspberry Pi 5 to running AI agent
# OS: Raspberry Pi OS 64-bit (Debian 13 trixie)
# Hardware: Pi 5 8GB, boots from microSD; data/ and ollama/ on NVMe
set -e

echo "=== Setting up local AI agent on Raspberry Pi 5 ==="

# ── System packages ──────────────────────────────────────────────────────────
sudo apt-get update
sudo apt-get install -y curl git

# ── Docker ───────────────────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
fi

# ── uv + venv in the repo ────────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync   # creates .venv from pyproject.toml

# ── Start services ───────────────────────────────────────────────────────────
echo "Starting VectorAI DB and Ollama..."
# VectorAI runs as uid 999; a root-owned bind mount crash-loops it
mkdir -p data && sudo chown 999:999 data
sudo docker compose up -d

echo "Waiting for VectorAI DB to be ready..."
until curl -sf http://localhost:6573/ > /dev/null 2>&1; do
    sleep 3
    echo "  ...waiting"
done
echo "VectorAI DB ready."

# ── Create agent_memory collection ───────────────────────────────────────────
# 768-dim matches nomic-embed-text output
echo "Creating agent_memory collection (768-dim, cosine, HNSW)..."
curl -s -X PUT http://localhost:6573/collections/agent_memory \
    -H "Content-Type: application/json" \
    -d '{"vectors": {"size": 768, "distance": "Cosine"}}'
echo ""

# ── Pull models via Ollama ───────────────────────────────────────────────────
echo "Pulling nomic-embed-text (274 MB)..."
sudo docker exec ollama ollama pull nomic-embed-text

echo "Pulling Qwen 2.5 1.5B (~1.1 GB)..."
sudo docker exec ollama ollama pull qwen2.5:1.5b

# ── Smoke tests ──────────────────────────────────────────────────────────────
echo ""
echo "=== Smoke tests ==="

echo "VectorAI DB collections:"
curl -s http://localhost:6573/collections | python3 -c "import sys,json; d=json.load(sys.stdin); print([c['name'] for c in d.get('result',{}).get('collections',[])])"

echo "Ollama models:"
sudo docker exec ollama ollama list

echo ""
echo "=== Setup complete ==="
echo "Log out and back in once so docker works without sudo, then:"
echo "  uv run agent.py"
