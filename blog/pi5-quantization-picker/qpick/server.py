"""Start llama.cpp's llama-server for one GGUF file, send chat requests, and stop it."""
import re
import subprocess
import time
from pathlib import Path

import requests

PORT = 8080                          # llama-server listens here, on the Pi only
URL = f"http://127.0.0.1:{PORT}"
THINK = re.compile(r"<think>.*?</think>", re.S)   # a model's hidden reasoning block, if one slips through


class LlamaServer:
    """One running copy of llama-server with one model loaded: start it, send it prompts, stop it."""

    def __init__(self, binary: Path, model: Path, ctx: int = 4096, threads: int = 4, log: Path | None = None):
        """Build the command line for one model file; the server is not started until start() is called."""
        # The command line: the model file, a 4,096-token context, 4 CPU threads, and no GPU layers.
        # -np 1 gives the server one request slot. Left on auto, it opens 4 slots and reserves memory
        # for 4 x 4,096 tokens of context, which more than doubles a small model's footprint.
        # --jinja makes the server use each model's own chat template.
        # --reasoning off turns thinking mode off for every model. Left on auto, the server switches thinking
        # on for any model whose template supports it (Gemma 4 does), and the model spends its whole answer
        # budget thinking, so the reply comes back empty.
        self.cmd = [str(binary), "-m", str(model), "-c", str(ctx), "-np", "1", "-t", str(threads), "-ngl", "0",
                    "--host", "127.0.0.1", "--port", str(PORT), "--jinja", "--reasoning", "off"]
        self.log = log
        self.proc = None

    def start(self) -> None:
        """Launch llama-server in the background, writing its output to the log file."""
        out = open(self.log, "w") if self.log else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.cmd, stdout=out, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout: float) -> bool:
        """True once the model is loaded; False if the process dies (out of memory) or the timeout passes."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:      # the process has exited, usually because memory ran out
                return False
            try:
                # /health returns 200 once the model is fully loaded and ready for requests
                if requests.get(f"{URL}/health", timeout=2).status_code == 200:
                    return True
            except requests.RequestException:     # not listening yet; keep waiting
                pass
            time.sleep(0.5)
        return False

    def alive(self) -> bool:
        """True while the llama-server process is still running."""
        return self.proc is not None and self.proc.poll() is None

    def chat(self, messages: list[dict], max_tokens: int, timeout: float, template_kwargs: dict | None = None) -> str:
        """Send one prompt and return the model's answer."""
        # temperature 0 and a fixed seed give the same answer every time, so runs can be compared
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 1}
        if template_kwargs:                       # e.g. Qwen3's enable_thinking: false
            body["chat_template_kwargs"] = template_kwargs
        r = requests.post(f"{URL}/v1/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"].get("content") or ""
        return THINK.sub("", text).strip()        # score only the answer, never the reasoning

    def stop(self) -> None:
        """Ask llama-server to exit, and force it if it hasn't after 30 seconds."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
