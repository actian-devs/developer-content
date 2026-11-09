"""Measure every model at every quantization level on the Pi 5, with the retrieval stack running.

For each model and level: download the GGUF file, start llama-server, record peak memory while it
loads and answers the fixed prompt set, score every answer, stop the server, and delete the file
(unless --keep). Results go to results/<model>/<level>.json; a finished level is skipped on rerun.

  uv run run.py                          # every model, every level
  uv run run.py --models qwen3-4b        # one model
  uv run run.py --levels Q4_K_M Q8_0     # some levels
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from qpick import corpus, prompts, stack
from qpick.memory import GIB, Sampler, meminfo
from qpick.models import LEVELS, MODELS, by_key, model_file
from qpick.server import LlamaServer

MAX_TOKENS = {"rag": 96, "json": 160, "instruction": 96}
SWAP_LIMIT_GIB = 0.25    # more swap growth than this and the level does not fit next to the stack
PROMPT_TIMEOUT = 180     # seconds; a swap-bound model blows past this
MAX_TIMEOUTS = 3         # stop a level after this many timed-out prompts


def _cmd(*args) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def board() -> dict:
    model = Path("/proc/device-tree/model")
    osr = Path("/etc/os-release")
    os_name = next((ln.split("=", 1)[1].strip('"') for ln in osr.read_text().splitlines()
                    if ln.startswith("PRETTY_NAME=")), None) if osr.exists() else platform.platform()
    return {"board": model.read_text().strip("\x00") if model.exists() else platform.node(),
            "ram_gib": round(meminfo()["total"] / GIB, 2), "swap_total_gib": round(meminfo()["swap_total"] / GIB, 2),
            "swap_devices": _cmd("swapon", "--show", "--noheadings"),   # swapfile or zram
            "os": os_name, "kernel": platform.release(),
            "root_device": _cmd("findmnt", "-no", "SOURCE", "/"), "repo_device": _cmd("findmnt", "-no", "SOURCE", "-T", ".")}


def thermals() -> dict:
    """Pi temperature and throttle flags (0x0 means it never throttled or ran under-voltage)."""
    temp = _cmd("vcgencmd", "measure_temp")
    return {"temp": temp.split("=")[-1] if temp else None, "throttled": _cmd("vcgencmd", "get_throttled")}


def task_list(recs, retrieved, rag_qs):
    tasks = []
    for q, hit in zip(rag_qs, retrieved):
        tasks.append({"kind": "rag", "q": q["q"], "expect": q["expect"], "retrieved_ids": hit["ids"],
                      "messages": prompts.rag_prompt(q["q"], hit["chunks"])})
    tasks += prompts.json_tasks(recs)
    tasks += prompts.instruction_tasks()
    return tasks


def score(task, answer, recs):
    if task["kind"] == "rag":
        return prompts.check_rag(answer, task["expect"])
    if task["kind"] == "json":
        return prompts.check_json(answer, recs[task["rec"]])
    return prompts.check_instruction(answer, task["idx"])


def run_level(model, level, args, tasks, recs, out_path):
    file = model_file(model, level)
    t0 = time.monotonic()
    path = Path(hf_hub_download(model.repo, file, local_dir=args.models_dir))
    result = {"model": model.name, "key": model.key, "level": level, "file": file, "repo": model.repo,
              "file_gib": round(path.stat().st_size / GIB, 3), "download_s": round(time.monotonic() - t0, 1),
              "ctx": args.ctx, "board": board(), "thermals_before": thermals()}
    log = out_path.with_suffix(".server.log")
    server = LlamaServer(args.llama_server, path, ctx=args.ctx, threads=args.threads, log=log)
    stack.warm_embedder()   # keep the embedding model loaded; Ollama unloads idle models
    server.start()
    answers, timeouts = [], 0
    with Sampler(pid=server.proc.pid) as mem:
        t0 = time.monotonic()
        loaded = server.wait_ready(args.load_timeout)
        result["load_s"] = round(time.monotonic() - t0, 1)
        result["loaded"] = loaded
        if loaded:
            for task in tasks:
                if not server.alive():
                    break
                t1 = time.monotonic()
                try:
                    answer = server.chat(task["messages"], MAX_TOKENS[task["kind"]], PROMPT_TIMEOUT, model.template_kwargs)
                    ok, error = score(task, answer, recs), None
                except Exception as e:   # a timeout or a server that died mid-answer
                    answer, ok, error = "", False, type(e).__name__
                    timeouts += error in ("ReadTimeout", "Timeout")
                answers.append({k: task[k] for k in ("kind", "q", "expect", "rec", "idx") if k in task} |
                               {"answer": answer, "pass": ok, "error": error, "seconds": round(time.monotonic() - t1, 1)})
                if timeouts >= MAX_TIMEOUTS:
                    break
    result["survived"] = server.alive()
    result["thermals_after"] = thermals()
    server.stop()
    result["memory"] = mem.report()
    result["timeouts"] = timeouts
    result["answers"] = answers
    done = len(answers) == len(tasks) and all(a["error"] is None for a in answers)
    for kind in ("rag", "json", "instruction"):
        sub = [a for a in answers if a["kind"] == kind]
        result[f"{kind}_pass"] = sum(a["pass"] for a in sub)
        result[f"{kind}_total"] = sum(t["kind"] == kind for t in tasks)
    result["quality"] = round(sum(a["pass"] for a in answers) / len(tasks), 3) if done else None
    if not loaded:
        result["fit"] = "fails to load"
    elif not done:
        result["fit"] = "swap-bound" if timeouts else "crashed"
    elif result["memory"]["swap_growth_gib"] > SWAP_LIMIT_GIB:
        result["fit"] = "swaps"
    else:
        result["fit"] = "fits"
    out_path.write_text(json.dumps(result, indent=1))
    if not args.keep:
        path.unlink(missing_ok=True)
    print(f"  {model.name} {level}: {result['fit']}, file {result['file_gib']} GiB, "
          f"peak used {result['memory']['peak_used_gib']} GiB, swap +{result['memory']['swap_growth_gib']} GiB, "
          f"quality {result['quality']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[m.key for m in MODELS])
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--llama-server", type=Path, default=Path("llama.cpp/build/bin/llama-server"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--load-timeout", type=float, default=900)
    ap.add_argument("--keep", action="store_true", help="keep each GGUF file after testing it")
    ap.add_argument("--redo", action="store_true", help="rerun levels that already have results")
    args = ap.parse_args()

    cache = Path("results/cache")
    recs = corpus.records()
    rag_qs = corpus.rag_questions(recs)
    print(f"Indexing {len(recs)} catalog chunks in VectorAI DB...")
    stack.build_index(recs, cache)
    retrieved = stack.retrieve([q["q"] for q in rag_qs], cache)
    hit = sum(q["rec"] in r["ids"] for q, r in zip(rag_qs, retrieved) if q["rec"] is not None)
    print(f"Retrieval: the right chunk is in the top {stack.TOP_K} for {hit}/16 answerable questions")
    tasks = task_list(recs, retrieved, rag_qs)

    with Sampler() as idle:
        time.sleep(3)
    versions = {"llama_server": _cmd(str(args.llama_server), "--version"),
                "docker_images": _cmd("docker", "ps", "--format", "{{.Names}} {{.Image}}"),
                "ollama": _cmd("docker", "exec", "ollama", "ollama", "--version")}
    (cache / "idle.json").write_text(json.dumps({"board": board(), "versions": versions, "thermals": thermals(),
                                                 "retrieval_top3_hits": f"{hit}/16", **idle.report()}, indent=1))
    print(f"Stack idle: board uses {idle.report()['peak_used_gib']} GiB before any LLM loads")

    for key in args.models:
        model = by_key(key)
        for level in args.levels:
            out = Path("results") / model.key / f"{level}.json"
            if out.exists() and not args.redo:
                print(f"  {model.name} {level}: done, skipping")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            run_level(model, level, args, tasks, recs, out)


if __name__ == "__main__":
    main()
