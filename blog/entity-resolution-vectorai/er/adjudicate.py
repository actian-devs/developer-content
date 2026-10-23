"""Gemini 4 Argon on the ambiguous band: a match / no-match verdict with a one-sentence reason.

Every verdict is cached in a JSONL log keyed by pair, model, and prompt version, so a rerun
never pays twice for the same pair and every decision stays auditable.
"""
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

PROMPT_VERSION = "v1"
PROMPT = """You are resolving product records from two catalogues.
Decide whether Record A and Record B describe the same product: same model, size, and variant.
Answer from the records only. Give a one-sentence reason.

Record A: {a}
Record B: {b}"""


class Verdict(BaseModel):
    match: bool
    reason: str


class Adjudicator:
    def __init__(self, model: str, log_path: Path, workers: int = 8, retries: int = 8,
                 backend: str = "ai-studio", thinking_level: str | None = "LOW"):
        self.backend = backend
        self.types, self.thinking_level = None, thinking_level
        if backend == "openrouter":
            # OpenRouter resells Google's Gemini 4 Argon (served by Google) on prepaid credit; key in OPENROUTER_API_KEY
            self.or_key = os.environ["OPENROUTER_API_KEY"]
            self.or_model = model if "/" in model else f"google/{model}"
        else:
            self._init_google(backend)
        # the cache key includes the thinking level, since it can change the verdict
        self.model_tag = f"{model}@{thinking_level or 'default'}"
        self.model, self.log_path, self.workers, self.retries = model, log_path, workers, retries
        self.cache = {}
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[(rec["left"], rec["right"], rec["model"], rec["prompt_version"])] = rec

    def _init_google(self, backend: str) -> None:
        from google import genai
        from google.genai import types

        self.types = types
        if backend == "agent-platform":
            # Google Cloud (Gemini Enterprise Agent Platform, formerly Vertex AI). A Google Cloud API key
            # (GOOGLE_CLOUD_API_KEY) is simplest; otherwise Application Default Credentials with
            # GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION (default "global").
            key = os.environ.get("GOOGLE_CLOUD_API_KEY")
            self.client = (genai.Client(enterprise=True, api_key=key) if key else
                           genai.Client(enterprise=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                                        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global")))
        else:
            self.client = genai.Client()   # AI Studio key from GEMINI_API_KEY

    def _call_google(self, prompt: str):
        thinking = (self.types.ThinkingConfig(thinking_level=self.thinking_level)
                    if self.thinking_level else None)   # LOW keeps billed thinking tokens down
        config = self.types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Verdict, temperature=0,
            thinking_config=thinking)
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        usage = resp.usage_metadata
        # thinking tokens are billed at the output rate, so count them with the visible answer
        out_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return Verdict.model_validate_json(resp.text), usage.prompt_token_count or 0, out_tokens, None

    def _call_openrouter(self, prompt: str):
        import requests

        body = {
            "model": self.or_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": {
                "type": "object", "additionalProperties": False, "required": ["match", "reason"],
                "properties": {"match": {"type": "boolean"}, "reason": {"type": "string"}}}}},
            "usage": {"include": True},   # OpenRouter returns the exact charge for the call
        }
        if self.thinking_level:
            body["reasoning"] = {"effort": self.thinking_level.lower()}
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=180,
                          headers={"Authorization": f"Bearer {self.or_key}"})
        r.raise_for_status()
        d = r.json()
        u = d["usage"]   # completion_tokens include reasoning tokens
        verdict = Verdict.model_validate_json(d["choices"][0]["message"]["content"])
        return verdict, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cost")

    def _call(self, a_text: str, b_text: str):
        """One verdict: (Verdict, input tokens, output tokens incl. thinking, exact cost in USD or None)."""
        prompt = PROMPT.format(a=a_text, b=b_text)
        call = self._call_openrouter if self.backend == "openrouter" else self._call_google
        for attempt in range(self.retries):
            try:
                return call(prompt)
            except Exception as e:   # rate limits and transient errors: back off and retry
                if attempt == self.retries - 1:
                    raise
                # honour the server's Retry-After (seconds) on a 429, else back off exponentially, capped at 60 s
                resp = getattr(e, "response", None)
                retry_after = getattr(resp, "headers", {}).get("Retry-After") if resp is not None else None
                wait = float(retry_after) if retry_after and str(retry_after).replace(".", "").isdigit() \
                    else min(60.0, 2.0 * 2 ** attempt)
                time.sleep(wait + random.uniform(0, 1))

    def run(self, band: list[tuple[str, str]], text_left: dict, text_right: dict,
            budget_left_usd: float = float("inf"), price_in: float = 2.0, price_out: float = 10.0) -> dict:
        """Return {(left_id, right_id): verdict record} for the band pairs decided so far.

        Stops sending new requests once this run's spend reaches `budget_left_usd`; the overshoot is
        at most one batch of `workers` calls. Pairs left undecided stay out of the result (and out of
        the cache), so a later run with more budget picks them up without paying twice.
        """
        key = lambda p: (p[0], p[1], self.model_tag, PROMPT_VERSION)
        todo = [p for p in band if key(p) not in self.cache]

        failed = []

        def work(pair):
            t0 = time.perf_counter()
            try:
                verdict, t_in, t_out, cost = self._call(text_left[pair[0]], text_right[pair[1]])
            except Exception as e:   # still failing after all retries: skip it, a rerun picks it up
                failed.append((pair, repr(e)[:200]))
                return None
            return {"left": pair[0], "right": pair[1], "model": self.model_tag, "prompt_version": PROMPT_VERSION,
                    "backend": self.backend, "match": verdict.match, "reason": verdict.reason,
                    "tokens_in": t_in, "tokens_out": t_out, "cost_usd": cost,
                    "seconds": round(time.perf_counter() - t0, 3)}

        spent, done = 0.0, 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log, ThreadPoolExecutor(self.workers) as pool:
            for start in range(0, len(todo), self.workers):
                if spent >= budget_left_usd:
                    print(f"  Budget reached (${spent:.4f} this run): stopped with "
                          f"{len(todo) - done} pairs undecided. Rerun with more budget to finish.")
                    break
                for rec in pool.map(work, todo[start:start + self.workers]):
                    if rec is None:
                        continue
                    self.cache[key((rec["left"], rec["right"]))] = rec
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    spent += record_cost(rec, price_in, price_out)
                    done += 1
                if done % 40 < self.workers or start + self.workers >= len(todo):
                    print(f"  Argon: {done}/{len(todo)} new verdicts, ${spent:.4f} spent this run"
                          + (f", {len(failed)} failed (will retry on rerun)" if failed else ""))
        if failed:
            print(f"  {len(failed)} pairs failed after retries, e.g. {failed[0][1]}")
        return {p: self.cache[key(p)] for p in band if key(p) in self.cache}


def spent_so_far(results_dir: Path, price_in: float, price_out: float) -> float:
    """Total Argon spend recorded in every argon_verdicts.jsonl under results/ (all datasets, all runs)."""
    total = 0.0
    for log in results_dir.glob("*/argon_verdicts.jsonl"):
        for line in log.read_text(encoding="utf-8").splitlines():
            total += record_cost(json.loads(line), price_in, price_out)
    return total


def record_cost(rec: dict, price_in: float, price_out: float) -> float:
    """The call's exact charge when the backend reported one (OpenRouter), else tokens x list price."""
    if rec.get("cost_usd") is not None:
        return float(rec["cost_usd"])
    return rec["tokens_in"] / 1e6 * price_in + rec["tokens_out"] / 1e6 * price_out
