"""Write a few user-scoped memories and print what mem0 pulled out of them."""

from memory import USER, build_memory

TURNS = [
    ("I moved the project off Chroma in March, the concurrency was killing us.",
     "Understood - what did you move to?"),
    ("Postgres with pgvector. It's been fine so far.",
     "Good choice under ten million vectors."),
    ("Keep answers short when you reply to me, I skim.",
     "Will do."),
    ("I'm on the platform team, so anything you suggest has to run on our own hardware.",
     "Noted - self-hosted only."),
]

m = build_memory()

for user_message, reply in TURNS:
    result = m.add(
        [{"role": "user", "content": user_message},
         {"role": "assistant", "content": reply}],
        user_id=USER,
    )
    for fact in result.get("results", []):
        print("  +", fact.get("memory"))
