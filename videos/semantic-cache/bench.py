"""Fire a batch of real-shaped traffic at the endpoint and read off the hit rate."""

import json
import pathlib
import statistics

import httpx

ENDPOINT = "http://localhost:8000/ask"
QUESTIONS = pathlib.Path(__file__).parent / "data" / "batch.json"


def main():
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    hits, misses = [], []

    for i, question in enumerate(questions, 1):
        response = httpx.post(ENDPOINT, json={"question": question}, timeout=180)
        if response.status_code != 200:
            raise SystemExit(f"{ENDPOINT} returned {response.status_code}: {response.text[:120]}")
        body = response.json()
        (hits if body["cached"] else misses).append(body["ms"])
        rate = len(hits) / i
        print(f"{i:3d}/{len(questions)}  {'HIT ' if body['cached'] else 'miss'}  {body['ms']:8.1f} ms   hit rate {rate:5.0%}    Q: {question}")

    print("\n            count   median ms")
    print(f"  hits    {len(hits):5d}   {statistics.median(hits) if hits else 0:9.1f}")
    print(f"  misses  {len(misses):5d}   {statistics.median(misses) if misses else 0:9.1f}")
    print(f"\n  model calls avoided: {len(hits)}")


if __name__ == "__main__":
    main()
