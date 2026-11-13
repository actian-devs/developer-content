"""
VectorAI DB benchmark: memory and disk at 1k, 10k and 100k vectors (float32 vs int8).

NOTE: above 5,000 vectors needs the trial license (cap raised to 1M). On the
Community Edition cap the run will stop at 5,000.

One container restart per precision. Inside one collection we ingest in steps,
measuring memory and disk at every checkpoint, then DELETE the collection.
A straight-line fit separates the fixed cost from the cost per added vector:
    growth(N) = fixed + slope * N

Setup:
    pip install actian-vectorai numpy
    caffeinate -i env CONTAINER=vectorai python bench_model.py     # macOS
    (optional) DATA_DIR=<container-side path>  DIM=384
"""
import json, os, statistics, subprocess, time

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

import numpy as np
from actian_vectorai import VectorAIClient, VectorParams, Distance, PointStruct
from actian_vectorai.models.collections import QuantizationConfig, ScalarQuantization
from actian_vectorai.models.enums import QuantizationType

SERVER = os.getenv("SERVER", "localhost:6574")
CONTAINER = os.getenv("CONTAINER", "vectorai")
DATA_DIR = os.getenv("DATA_DIR", "")               # container-side; empty = auto-detect mounts
DIM = int(os.getenv("DIM", "384"))
CHECKPOINTS = [1000, 10_000, 100_000]
ARTICLE_SIZES = [1000, 10_000, 100_000]
PRECISIONS = ["float32", "int8"]
COLLECTION = "bench"
BATCH = 1000
BYTES = {"float32": 4, "int8": 1}                  # B in N*D*B
MIB = 1024 ** 2
MEM_SOURCE = None


# ---------- measurement ----------
def _anon_mib():
    """Process memory (anon/rss) from the container's cgroup; excludes page cache."""
    for path in ("/sys/fs/cgroup/memory.stat", "/sys/fs/cgroup/memory/memory.stat"):
        r = subprocess.run(["docker", "exec", CONTAINER, "cat", path],
                           capture_output=True, text=True)
        if r.returncode == 0:
            s = {k: v for k, v, *_ in (l.split() for l in r.stdout.splitlines() if len(l.split()) >= 2)}
            key = "anon" if "anon" in s else "rss"
            if key in s:
                return int(s[key]) / MIB
    return None


def _stats_mib():
    out = subprocess.check_output(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER],
        text=True).split("/")[0].strip()
    to_mib = {"GiB": 1024, "MiB": 1, "KiB": 1 / 1024, "B": 1 / MIB}
    for unit in ("GiB", "MiB", "KiB", "B"):
        if out.endswith(unit):
            return float(out[: -len(unit)]) * to_mib[unit]
    raise ValueError(f"unrecognised docker stats value: {out}")


def mem_mib() -> float:
    global MEM_SOURCE
    if MEM_SOURCE is None:
        MEM_SOURCE = "cgroup anon" if _anon_mib() is not None else "docker stats"
    read = _anon_mib if MEM_SOURCE == "cgroup anon" else _stats_mib
    samples = []
    for _ in range(3):
        samples.append(read()); time.sleep(1)
    return statistics.median(samples)


def data_dirs():
    if DATA_DIR:
        return [DATA_DIR]
    out = subprocess.check_output(
        ["docker", "inspect", CONTAINER, "--format", "{{json .Mounts}}"], text=True)
    return [m["Destination"] for m in json.loads(out)]


def disk_mib() -> float:
    total_kb, found = 0, False
    for d in data_dirs():
        r = subprocess.run(["docker", "exec", CONTAINER, "du", "-sk", d],
                           capture_output=True, text=True)
        if r.returncode == 0:
            total_kb += int(r.stdout.split()[0]); found = True
    if not found:
        raise RuntimeError("No readable data dir. Set DATA_DIR to the container-side path.")
    return total_kb / 1024


# ---------- benchmark ----------
def make_collection(client, precision):
    kwargs = {}
    if precision == "int8":
        kwargs["quantization_config"] = QuantizationConfig(
            scalar=ScalarQuantization(type=QuantizationType.Int8, quantile=0.99, always_ram=True))
    client.collections.create(
        COLLECTION, vectors_config=VectorParams(size=DIM, distance=Distance.Cosine), **kwargs)


def wait_ready():
    for _ in range(90):
        try:
            with VectorAIClient(SERVER) as c:
                c.health_check()
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not come back after restart")


def run_precision(precision, rng):
    with VectorAIClient(SERVER) as c:                  # clean slate, then fresh process
        if c.collections.exists(COLLECTION):
            c.collections.delete(COLLECTION)
    subprocess.run(["docker", "restart", CONTAINER], check=True, capture_output=True)
    wait_ready()
    time.sleep(5)
    base_mem, base_disk = mem_mib(), disk_mib()        # idle server, no collection
    pts = []

    def record(n):
        pts.append({"n": n, "mem_mib": mem_mib() - base_mem, "disk_mib": disk_mib() - base_disk})
        print(f"   N={n:>7,}  mem +{pts[-1]['mem_mib']:.1f} MiB  disk +{pts[-1]['disk_mib']:.1f} MiB")

    with VectorAIClient(SERVER) as client:
        make_collection(client, precision)
        time.sleep(3)
        record(0)                                      # empty collection = fixed cost
        done = 0
        for cp in CHECKPOINTS:
            t0, before = time.perf_counter(), done
            for s in range(done, cp, BATCH):
                e = min(s + BATCH, cp)
                block = rng.standard_normal((e - s, DIM), dtype=np.float32)
                client.points.upsert(COLLECTION, [
                    PointStruct(id=s + j, vector=block[j].tolist(), payload={"i": s + j})
                    for j in range(e - s)])
                if (s // BATCH) % 20 == 0:
                    print(f"      ...{e:,}/{cp:,}", flush=True)
            done = cp
            client.vde.flush(COLLECTION)
            secs = time.perf_counter() - t0
            print(f"   ingested {cp - before:,} in {secs:.0f}s ({(cp - before) / secs:,.0f} vec/s, incl. flush)")
            count = client.vde.get_vector_count(COLLECTION)
            assert count == cp, f"expected {cp}, got {count}"
            time.sleep(5 if cp < 50_000 else 30)       # let index build / memory settle
            record(cp)
        client.collections.delete(COLLECTION)
    return {"base_mem_mib": base_mem, "base_disk_mib": base_disk, "points": pts}


# ---------- analysis ----------
def fit(points, key):
    pts = [p for p in points if p["n"] > 0]
    x = np.array([p["n"] for p in pts], float)
    y = np.array([p[key] for p in pts], float)
    slope, icpt = np.polyfit(x, y, 1)
    ss, tot = ((y - (slope * x + icpt)) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return slope, icpt, (1 - ss / tot if tot else 1.0)


def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def html_table(head, rows):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def chart(results, fits):
    W, H, L, R, T, Bm = 720, 400, 64, 16, 16, 48
    xmax = max(CHECKPOINTS)
    ymax = max(p["mem_mib"] for r in results.values() for p in r["points"]) * 1.1
    X = lambda n: L + (W - L - R) * n / xmax
    Y = lambda v: H - Bm - (H - T - Bm) * v / ymax
    col = {"float32": "#2563eb", "int8": "#d97706"}
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" font-size="12" fill="currentColor">']
    for i in range(6):
        v = ymax * i / 5
        s.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="currentColor" opacity=".15"/>'
                 f'<text x="{L-6}" y="{Y(v)+4:.1f}" text-anchor="end">{v:.0f}</text>')
    for n in range(0, xmax + 1, xmax // 5):
        s.append(f'<text x="{X(n):.1f}" y="{H-Bm+18}" text-anchor="middle">{n:,}</text>')
    mid = (T + H - Bm) / 2
    s.append(f'<text x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle">vectors stored</text>')
    s.append(f'<text x="14" y="{mid}" text-anchor="middle" transform="rotate(-90 14 {mid})">memory growth (MiB)</text>')
    ly = T + 14
    for prec, res in results.items():
        c = col.get(prec, "#888")
        ms, mi = fits[prec]
        s.append(f'<line x1="{X(0):.1f}" y1="{Y(mi):.1f}" x2="{X(xmax):.1f}" y2="{Y(ms*xmax+mi):.1f}" stroke="{c}" stroke-width="2"/>')
        for p in res["points"]:
            s.append(f'<circle cx="{X(p["n"]):.1f}" cy="{Y(p["mem_mib"]):.1f}" r="4" fill="{c}"/>')
        s.append(f'<rect x="{L+10}" y="{ly-9}" width="12" height="12" fill="{c}"/>'
                 f'<text x="{L+28}" y="{ly+2}">{prec.capitalize()}: measured (dots) and linear fit (line)</text>')
        ly += 18
    return "".join(s) + "</svg>"


def analyse(results):
    model, pts_rows, article = [], [], []
    fits = {}
    for pi, (prec, res) in enumerate(results.items()):
        pts, b = res["points"], BYTES[prec]
        raw_per_vec = DIM * b
        ms, mi, mr = fit(pts, "mem_mib")
        ds, _, _ = fit(pts, "disk_mib")
        fits[prec] = (ms, mi)
        n0 = next((p["mem_mib"] for p in pts if p["n"] == 0), float("nan"))
        model.append([prec.capitalize(), f"{mi:.1f}", f"{n0:.1f}", f"{ms*MIB:,.0f}", f"{ds*MIB:,.0f}",
                      f"{raw_per_vec:,}", f"{ms*MIB/raw_per_vec:.1f}x", f"{mr:.3f}"])
        for p in pts:
            if p["n"] in ARTICLE_SIZES:
                article.append((p["n"], pi, [f"{p['n']:,}", prec.capitalize(),
                                f"{p['n']*DIM*b/MIB:.2f} MiB", f"{p['mem_mib']:.1f} MiB",
                                f"{p['disk_mib']:.1f} MiB"]))
            if p["n"] > 0:
                pts_rows.append([f"{p['n']:,}", prec.capitalize(),
                                 f"{p['mem_mib']:.1f}", f"{p['mem_mib'] - mi:.2f}"])
    article = [r for _, _, r in sorted(article, key=lambda t: (t[0], t[1]))]

    h_art = ["Vectors", "Precision", "Raw vectors (N·D·B)", "Measured memory", "Measured disk"]
    h_mod = ["Precision", "Fixed cost (fit, MiB)", "Empty collection (MiB)", "Memory B/vector",
             "Disk B/vector", "Raw B/vector (D·B)", "Memory ÷ raw", "R²"]
    h_pts = ["Vectors", "Precision", "Measured (MiB)", "Measured − fixed (MiB)"]

    md = (f"## Article table\n\n{md_table(h_art, article)}\n\n"
          f"## Model fit: growth(N) = fixed + slope·N  (dim={DIM}, memory source: {MEM_SOURCE})\n\n"
          f"{md_table(h_mod, model)}\n\n## Per checkpoint\n\n{md_table(h_pts, pts_rows)}\n")
    open("benchmark_model.md", "w").write(md)

    css = ("body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}"
           "table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem;font-size:14px}"
           "th,td{border:1px solid #8884;padding:.4rem .6rem;text-align:right}"
           "th{background:#8882}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}"
           "code{background:#8882;padding:2px 5px;border-radius:4px}")
    html = ('<!doctype html><meta charset="utf-8"><title>VectorAI DB: measured memory and disk</title>'
            f"<style>{css}</style><h2>VectorAI DB: measured memory and disk</h2>"
            f"<p>Dim {DIM}, cosine, default HNSW. Memory source: <b>{MEM_SOURCE}</b>. "
            "Values are growth over an idle, freshly restarted server. "
            "Raw = <code>N × D × B</code>; the fitted line is <code>fixed + slope × N</code>.</p>"
            f"{chart(results, fits)}<h3>Article table</h3>{html_table(h_art, article)}"
            f"<h3>Model fit</h3>{html_table(h_mod, model)}"
            f"<h3>Per checkpoint</h3>{html_table(h_pts, pts_rows)}")
    open("benchmark_model.html", "w").write(html)
    print("\n" + md)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    with VectorAIClient(SERVER) as c:
        print("Connected:", c.health_check())
    print("Measuring disk in:", data_dirs())
    results = {}
    for p in PRECISIONS:
        print(f"-> {p}")
        results[p] = run_precision(p, rng)
        json.dump(results, open("benchmark_model_results.json", "w"), indent=2)  # keep partial results
    analyse(results)
