"""The fixed prompt set and the automatic check for each prompt.

Three kinds of task, matching how readers use a small model next to a vector store:
  rag          answer from chunks retrieved from VectorAI DB (20 prompts)
  json         extract a record into JSON with given keys and types (10 prompts)
  instruction  follow a short, checkable instruction (10 prompts)
Every check is a pure function of the answer text, so a score never depends on a judge model.
"""
import json
import random
import re

from qpick.corpus import SEED

# Phrases that count as "this product is not in the catalog", for the 4 questions about products that don't exist
NOT_FOUND = re.compile(r"\b(not (in|found|mentioned|listed|included|provided|available)|no (information|record|mention)|"
                       r"(isn't|is not|doesn't|does not) (in|appear|exist|contain|mention|list)|cannot find|can't find|unknown)\b", re.I)

# The instruction every RAG prompt starts with: answer from the retrieved entries only, and admit it if they don't help
RAG_SYSTEM = ("Answer the question using only the catalog entries below. Answer in one sentence. "
              "If the entries do not contain the product, say it is not in the catalog.")


def _norm(s: str) -> str:
    """Lowercase, and turn the typographic minus sign into a plain hyphen, so "-20" matches either way."""
    return s.lower().replace("−", "-")


def _has(answer: str, value: str) -> bool:
    """A number must appear as a whole number (16.0 also matches 16, 70 does not match 170); text as a substring."""
    if re.fullmatch(r"-?\d+(\.\d+)?", value):
        forms = {value, value[:-2]} if value.endswith(".0") else {value}
        return any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d]|\.\d)", answer) for f in forms)
    return _norm(value) in answer


def check_rag(answer: str, expect: list[str] | None) -> bool:
    """Pass if the answer contains every expected value, or, for a product not in the catalog, says so."""
    if expect is None:
        return bool(NOT_FOUND.search(answer))
    a = _norm(answer)
    return all(_has(a, e) for e in expect)


def rag_prompt(question: str, chunks: list[str]) -> list[dict]:
    """Build a RAG prompt: the instruction, then the 3 retrieved catalog entries, then the question."""
    context = "\n".join(f"- {c}" for c in chunks)
    return [{"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"Catalog entries:\n{context}\n\nQuestion: {question}"}]


# ── JSON extraction ──────────────────────────────────────────────────────────

# The model gets one catalog entry and must return it as JSON with exactly these five keys and types
JSON_INSTRUCTION = ("Extract this catalog entry into a JSON object with exactly these keys: "
                    "\"name\" (string), \"part_number\" (string), \"max_load_kg\" (number), "
                    "\"warranty_months\" (integer), \"plant\" (string). Reply with the JSON object only.\n\n")


def _parse_json(answer: str):
    """Read the JSON out of the answer. A ```json code fence around it is allowed; any other text is not."""
    text = answer.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def check_json(answer: str, rec: dict) -> bool:
    """Pass only if the JSON parses, has exactly the five keys, uses the right types, and matches the record."""
    try:
        obj = _parse_json(answer)
    except (ValueError, TypeError):
        return False                     # not valid JSON at all
    if not isinstance(obj, dict) or set(obj) != {"name", "part_number", "max_load_kg", "warranty_months", "plant"}:
        return False                     # a missing, extra or misspelt key
    return (obj["name"] == rec["name"] and obj["part_number"] == rec["part"]
            and isinstance(obj["max_load_kg"], (int, float)) and abs(obj["max_load_kg"] - rec["load_kg"]) < 1e-6
            # warranty_months must be a real integer, not "24" in quotes and not true/false
            and isinstance(obj["warranty_months"], int) and not isinstance(obj["warranty_months"], bool)
            and obj["warranty_months"] == rec["warranty_months"] and obj["plant"] == rec["plant"])


def json_tasks(recs: list[dict]) -> list[dict]:
    """10 JSON prompts, each built from a catalog entry picked with a fixed seed (the same 10 every run)."""
    rng = random.Random(SEED + 2)
    return [{"kind": "json", "rec": r["id"],
             "messages": [{"role": "user", "content": JSON_INSTRUCTION + r["text"]}]}
            for r in rng.sample(recs, 10)]


# ── Short instructions ───────────────────────────────────────────────────────

def _words(a: str) -> list[str]:
    """Split an answer into words, for the word-count check."""
    return re.findall(r"[A-Za-z0-9'°.-]+", a)


def _bullets(a: str) -> list[str]:
    """The lines of an answer that start with a bullet (-, * or •)."""
    return [ln for ln in a.strip().splitlines() if re.match(r"^\s*[-*•]\s+\S", ln)]


# Each entry is (the instruction sent to the model, the check its answer must pass)
INSTRUCTIONS = [
    # exactly three bullet lines and nothing else
    ("List exactly three benefits of active cooling for a Raspberry Pi 5, as three lines that each start with \"- \". "
     "Write nothing else.",
     lambda a: len(_bullets(a)) == 3 and len([ln for ln in a.strip().splitlines() if ln.strip()]) == 3),
    # a word limit
    ("In no more than 15 words, explain what a vector database stores.",
     lambda a: 0 < len(_words(a)) <= 15),
    # lowercase only
    ("Reply in lowercase letters only: why do small language models run on edge devices?",
     lambda a: a.strip() != "" and a == a.lower()),
    # a one-word answer
    ("Answer with a single word, Yes or No: is 8 GB more than 4 GB?",
     lambda a: re.fullmatch(r"\W*yes\W*", a.strip(), re.I) is not None),
    # a letter the answer must avoid
    ("Write one sentence about NVMe storage that does not contain the letter \"e\".",
     lambda a: a.strip() != "" and "e" not in a.lower()),
    # arithmetic, number only
    ("Give the result of 17 multiplied by 23. Reply with the number only.",
     lambda a: re.fullmatch(r"\W*391\W*", a.strip()) is not None),
    # three real rainbow colours, in alphabetical order, comma-separated
    ("Name three colours of the rainbow, separated by commas, in alphabetical order, with nothing else.",
     lambda a: (lambda p: len(p) == 3 and p == sorted(p) and all(x in {"red", "orange", "yellow", "green", "blue",
                "indigo", "violet"} for x in p))([x.strip().strip(".").lower() for x in a.strip().split(",")])),
    # one line, one question mark
    ("Reply with a question that asks the user which Raspberry Pi model they own. End it with a question mark "
     "and write nothing else.",
     lambda a: a.strip().endswith("?") and a.strip().count("?") == 1 and len(a.strip().splitlines()) == 1),
    # copy a part number exactly
    ("Repeat this text exactly, with no other words: NR-7F2-418",
     lambda a: a.strip().strip("`\"' ") == "NR-7F2-418"),
    # reverse a word and capitalize it
    ("Write the word \"retrieval\" backwards, in uppercase, and nothing else.",
     lambda a: a.strip().strip(".\"'` ") == "LAVEIRTER"),
]


def instruction_tasks() -> list[dict]:
    """The 10 instruction prompts, each remembering its position so its check can be found later."""
    return [{"kind": "instruction", "idx": i, "messages": [{"role": "user", "content": text}]}
            for i, (text, _) in enumerate(INSTRUCTIONS)]


def check_instruction(answer: str, idx: int) -> bool:
    """Run the check that belongs to instruction number idx."""
    return bool(INSTRUCTIONS[idx][1](answer))
