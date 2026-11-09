"""Sample memory while a model loads and answers: the whole board, the LLM process, and each container.

Reads Linux's own counters, so it adds almost nothing to what it measures:
  /proc/meminfo                       board-wide used RAM (MemTotal - MemAvailable) and swap in use
  /proc/<pid>/status (VmRSS)          resident memory of the llama-server process
  cgroup v2 memory.current            memory charged to each Docker container (VectorAI DB, Ollama)
"""
import subprocess
import threading
import time
from pathlib import Path

GIB = 1024 ** 3


def meminfo() -> dict:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, val = line.split(":", 1)
        info[key] = int(val.split()[0]) * 1024   # kB to bytes
    return {"total": info["MemTotal"], "used": info["MemTotal"] - info["MemAvailable"],
            "swap_used": info["SwapTotal"] - info["SwapFree"], "swap_total": info["SwapTotal"]}


def rss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def container_cgroups(names: list[str]) -> dict[str, Path]:
    """memory.current for each running container, under the systemd or cgroupfs layout."""
    out = {}
    for name in names:
        try:
            cid = subprocess.run(["docker", "inspect", "-f", "{{.Id}}", name], capture_output=True, text=True,
                                 check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        for p in (Path(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/memory.current"),
                  Path(f"/sys/fs/cgroup/docker/{cid}/memory.current")):
            if p.exists():
                out[name] = p
                break
    return out


class Sampler:
    """Background thread recording the peak of every counter, every `interval` seconds."""

    def __init__(self, pid: int | None = None, containers: list[str] = ("vectorai", "ollama"), interval: float = 0.25):
        self.pid, self.interval = pid, interval
        self.cgroups = container_cgroups(list(containers))
        self.peak = {"used": 0, "swap_used": 0, "llm_rss": 0, **{c: 0 for c in self.cgroups}}
        self.start_swap = meminfo()["swap_used"]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> dict:
        m = meminfo()
        s = {"used": m["used"], "swap_used": m["swap_used"], "llm_rss": rss(self.pid) if self.pid else 0}
        for name, path in self.cgroups.items():
            try:
                s[name] = int(path.read_text())
            except (OSError, ValueError):
                s[name] = 0
        return s

    def _run(self):
        while not self._stop.is_set():
            for k, v in self.sample().items():
                self.peak[k] = max(self.peak[k], v)
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def report(self) -> dict:
        return {**{f"peak_{k}_gib": round(v / GIB, 3) for k, v in self.peak.items()},
                "swap_growth_gib": round(max(0, self.peak["swap_used"] - self.start_swap) / GIB, 3)}
