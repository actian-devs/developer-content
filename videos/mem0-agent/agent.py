"""One agent turn, with memory either side of it: search before, add after."""

import sys

import ollama

from memory import LLM_MODEL, USER, build_memory


def answer(question, remembered):
    system = "You are a helpful assistant."
    if remembered:
        system += "\n\nWhat you remember about this user:\n" + "\n".join(f"- {r}" for r in remembered)
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": question}],
    )
    return response["message"]["content"]


def main():
    m = build_memory()
    print("memory ready. ctrl-c to quit.\n")
    while True:
        question = input("you > ").strip()
        if not question:
            continue

        # before the model sees anything
        remembered = [h["memory"] for h in m.search(question, filters={"user_id": USER}).get("results", [])]
        reply = answer(question, remembered)
        print(f"\nagent > {reply}\n")

        # after it has answered - mem0 decides what in the turn is worth keeping
        m.add([{"role": "user", "content": question},
               {"role": "assistant", "content": reply}], user_id=USER)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)
