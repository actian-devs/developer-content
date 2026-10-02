"""Retrieval latency benchmark for VectorAI DB on Raspberry Pi 5.

Loads 1K / 10K / 100K vectors (768-dim, same size as nomic-embed-text), runs
100 timed top-k searches per corpus size, and writes p50/p95/p99 to CSV.

Vectors are synthetic (random, unit-normalized) so the 100K load finishes in
minutes instead of hours of on-device embedding. Random vectors are a harder
case for HNSW than real text embeddings, so treat these numbers as conservative.

The free Community edition caps the server at 5,000 vectors; without a licence
run it with --sizes 1000 2000 4000.

Usage: python benchmark.py [--sizes 1000 10000 100000] [--queries 100]
"""
import argparse
import csv
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from actian_vectorai import Distance, PointStruct, VectorAIClient, VectorParams

DIM = 768
TOP_K = 5
BATCH = 500
WARMUP = 10


def hardware() -> dict:
    def read(path: str) -> str:
        try:
            return Path(path).read_text().strip("\x00\n ")
        except OSError:
            return "unknown"

    def cmd(*args: str) -> str:
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    mem_kb = next((line.split()[1] for line in read("/proc/meminfo").splitlines() if line.startswith("MemTotal")), "0")
    # ask Docker where the vectorai container keeps its data, so the CSV names the drive
    # the database actually writes to (the NVMe), wherever this script was cloned
    data_dir = cmd("docker", "inspect", "-f", '{{range .Mounts}}{{if eq .Destination "/var/lib/actian-vectorai"}}'
                   '{{.Source}}{{end}}{{end}}', "vectorai") or str(Path(__file__).resolve().parent / "data")
    return {
        "board": read("/proc/device-tree/model"),
        "arch": platform.machine(),
        "kernel": platform.release(),
        "ram_gb": round(int(mem_kb) / 1024 / 1024, 1),
        "storage": cmd("findmnt", "-no", "SOURCE", "-T", data_dir) or "unknown",
        # vcgencmd prints "temp=48.3'C" / "throttled=0x0"; keep only the value
        "temp": cmd("vcgencmd", "measure_temp").split("=")[-1],
        "throttled": cmd("vcgencmd", "get_throttled").split("=")[-1],  # 0x0 means no throttling since boot
    }


def unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def load(db: VectorAIClient, name: str, n: int, rng: np.random.Generator) -> float:
    if db.collections.exists(name):
        db.collections.delete(name)
    db.collections.create(name, vectors_config=VectorParams(size=DIM, distance=Distance.Cosine))
    t0 = time.perf_counter()
    for start in range(0, n, BATCH):
        vecs = unit_vectors(min(BATCH, n - start), rng)
        db.points.upsert(name, [
            PointStruct(id=start + i, vector=v.tolist(), payload={"i": start + i})
            for i, v in enumerate(vecs)
        ])
    return time.perf_counter() - t0


def measure(db: VectorAIClient, name: str, queries: int, rng: np.random.Generator) -> list[float]:
    qs = unit_vectors(queries + WARMUP, rng).tolist()
    for q in qs[:WARMUP]:  # first searches load the collection into memory
        db.points.search(name, vector=q, limit=TOP_K)
    lat = []
    for q in qs[WARMUP:]:
        t0 = time.perf_counter()
        db.points.search(name, vector=q, limit=TOP_K)
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    # Community edition (no licence) is capped at 5K vectors: use --sizes 1000 2000 4000
    ap.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--host", default="localhost:6574")
    ap.add_argument("--out", default="results/benchmark.csv")
    ap.add_argument("--keep", action="store_true", help="keep benchmark collections afterwards")
    args = ap.parse_args()

    hw = hardware()
    print("Hardware:", ", ".join(f"{k}={v}" for k, v in hw.items()))
    rng = np.random.default_rng(42)
    rows = []

    with VectorAIClient(args.host) as db:
        for n in args.sizes:
            name = f"bench_{n}"
            print(f"\n== {n:,} vectors ==")
            load_s = load(db, name, n, rng)
            print(f"  loaded in {load_s:.1f} s ({n / load_s:,.0f} vectors/s)")
            lat = measure(db, name, args.queries, rng)
            row = {
                "vectors": n, "dim": DIM, "top_k": TOP_K, "queries": len(lat),
                "p50_ms": round(pct(lat, 50), 2), "p95_ms": round(pct(lat, 95), 2),
                "p99_ms": round(pct(lat, 99), 2), "mean_ms": round(statistics.mean(lat), 2),
                "max_ms": round(max(lat), 2), "load_s": round(load_s, 1),
            }
            rows.append(row)
            print(f"  p50 {row['p50_ms']} ms | p95 {row['p95_ms']} ms | p99 {row['p99_ms']} ms")
            if not args.keep:
                db.collections.delete(name)

    hw_after = hardware()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["board", "ram_gb", "storage", "temp_end", "throttled_end"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "board": hw["board"], "ram_gb": hw["ram_gb"], "storage": hw["storage"],
                        "temp_end": hw_after["temp"], "throttled_end": hw_after["throttled"]})
    print(f"\nWrote {out}. End state: temp={hw_after['temp']}, throttled={hw_after['throttled']}")


if __name__ == "__main__":
    main()
