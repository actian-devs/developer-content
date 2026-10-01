"""Local AI agent with semantic memory: Ollama (Qwen 2.5 1.5B) + Actian VectorAI DB."""
import os
import sys
import time

import requests
from actian_vectorai import Distance, PointStruct, VectorAIClient, VectorParams

OLLAMA = "http://localhost:11434"
VECTORAI = "localhost:6574"
CHAT_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768            # nomic-embed-text output size
COLLECTION = "agent_memory"
TOP_K = 3
MIN_SCORE = 0.6            # unrelated memories score ~0.3-0.55 with nomic-embed-text, related ones 0.7+


def embed(text: str) -> list[float]:
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embeddings"][0]


def chat(messages: list[dict]) -> str:
    # Ollama exposes an OpenAI-compatible endpoint, so any OpenAI client works here too
    r = requests.post(f"{OLLAMA}/v1/chat/completions", timeout=300,
                      json={"model": CHAT_MODEL, "messages": messages, "temperature": 0.3})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# Tell the model what it runs on, so "what hardware is this?" gets a real answer
MODEL_FILE = "/proc/device-tree/model"
BOARD = open(MODEL_FILE).read().strip("\x00\n ") if os.path.exists(MODEL_FILE) else "unknown board"
HARDWARE = f"{BOARD}, {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3:.0f} GB RAM"


def step(db: VectorAIClient, user_msg: str) -> str:
    # 1. Embed the incoming query on the Pi CPU
    t0 = time.perf_counter()
    qvec = embed(user_msg)
    t1 = time.perf_counter()

    # 2. Retrieve the closest past exchanges from VectorAI DB (HNSW index)
    hits = db.points.search(COLLECTION, vector=qvec, limit=TOP_K, with_payload=True)
    t2 = time.perf_counter()
    memories = [h.payload["text"] for h in hits if h.score >= MIN_SCORE]

    # 3. Inject retrieved memories into the system prompt, labelled as past notes
    #    so a small model doesn't mistake them for the current message
    context = "\n".join(f"- {m}" for m in memories) or "- (none)"
    messages = [
        {"role": "system", "content": f"You are a helpful assistant running fully offline on {HARDWARE} "
         f"with {CHAT_MODEL} via Ollama and Actian VectorAI DB for memory.\n"
         "Every message the user sends is saved to your memory automatically. If the user tells you "
         "a fact, confirm in one sentence that you will remember it.\n"
         "Answer in 1-3 sentences. Only use the notes below if they help answer the current message.\n\n"
         "Notes from earlier conversations:\n" + context},
        {"role": "user", "content": user_msg},
    ]

    # 4. Local inference through Ollama
    answer = chat(messages)
    t3 = time.perf_counter()

    # 5. Write the user's message back so future sessions can recall it. Only the user's
    #    side is stored: the facts come from the user, and storing the model's own replies
    #    would feed any wrong answer back into later prompts. The query vector is reused.
    db.points.upsert(COLLECTION, [PointStruct(
        id=time.time_ns() // 1000,  # microsecond timestamp as a unique integer ID
        vector=qvec,
        payload={"text": f"User said: {user_msg}", "ts": time.time()},
    )])
    t4 = time.perf_counter()

    print(f"  [embed {1000 * (t1 - t0):.0f} ms | retrieval {1000 * (t2 - t1):.1f} ms "
          f"({len(memories)} memories) | inference {t3 - t2:.1f} s | write-back {1000 * (t4 - t3):.0f} ms]")
    return answer


def main() -> None:
    with VectorAIClient(VECTORAI) as db:
        if db.collections.exists(COLLECTION):
            # VectorAI DB keeps collections on disk but only loads them when opened;
            # without this, search and count fail after every DB restart
            db.vde.open_collection(COLLECTION)
        else:
            db.collections.create(COLLECTION, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.Cosine))
        print(f"Agent ready on {HARDWARE} | LLM: {CHAT_MODEL} | "
              f"{db.points.count(COLLECTION)} memories in VectorAI DB. Ctrl+C to quit.")
        if len(sys.argv) > 1:  # one-shot mode: python agent.py "your question"
            print(step(db, " ".join(sys.argv[1:])))
            return
        try:  # Ctrl+C quits cleanly, even mid-answer
            while True:
                user_msg = input("\nyou> ").strip()
                if user_msg:
                    print(f"agent> {step(db, user_msg)}")
        except (EOFError, KeyboardInterrupt):
            print()


if __name__ == "__main__":
    main()
