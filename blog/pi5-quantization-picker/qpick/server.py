"""Start llama.cpp's llama-server for one GGUF file, send chat requests, and stop it."""
import re
import subprocess
import time
from pathlib import Path

import requests

PORT = 8080
URL = f"http://127.0.0.1:{PORT}"
THINK = re.compile(r"<think>.*?</think>", re.S)


class LlamaServer:
    def __init__(self, binary: Path, model: Path, ctx: int = 4096, threads: int = 4, log: Path | None = None):
        self.cmd = [str(binary), "-m", str(model), "-c", str(ctx), "-t", str(threads), "-ngl", "0",
                    "--host", "127.0.0.1", "--port", str(PORT), "--jinja"]
        self.log = log
        self.proc = None

    def start(self) -> None:
        out = open(self.log, "w") if self.log else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.cmd, stdout=out, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout: float) -> bool:
        """True once the model is loaded; False if the process dies (out of memory) or the timeout passes."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                if requests.get(f"{URL}/health", timeout=2).status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(0.5)
        return False

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def chat(self, messages: list[dict], max_tokens: int, timeout: float, template_kwargs: dict | None = None) -> str:
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 1}
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        r = requests.post(f"{URL}/v1/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"].get("content") or ""
        return THINK.sub("", text).strip()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
