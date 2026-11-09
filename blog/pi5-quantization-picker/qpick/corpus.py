"""A fictional parts catalog, so no model can answer from what it memorized in training.

Each record is one chunk of text with a unique product name and four facts. The RAG prompts
ask about records by name, retrieve the top chunks from VectorAI DB, and check the answer
against the record. The catalog is generated from a fixed seed, so every run builds the same one.
"""
import random

SEED = 7
N_RECORDS = 4000   # stays under VectorAI DB Community Edition's 5,000-vector cap

ADJ = ["Halcyon", "Brisk", "Cobalt", "Juniper", "Tundra", "Vesper", "Marlow", "Quill", "Saffron", "Orrin",
       "Talon", "Wren", "Corvid", "Ember", "Larkspur", "Nimbus", "Pellet", "Rowan", "Sable", "Thistle"]
NOUN = ["servo bracket", "gear housing", "sensor mount", "drive coupling", "cable gland", "motor plate",
        "bearing block", "fan shroud", "relay board", "hinge arm", "pulley hub", "clamp collar"]
PLANT = ["Leeds", "Porto", "Gdansk", "Tampere", "Lyon", "Brno", "Cork", "Graz"]


def records() -> list[dict]:
    rng = random.Random(SEED)
    seen, out = set(), []
    while len(out) < N_RECORDS:
        name = f"{rng.choice(ADJ)} {rng.randint(10, 99)} {rng.choice(NOUN)}"
        if name in seen:
            continue
        seen.add(name)
        rec = {
            "id": len(out),
            "name": name,
            "part": f"NR-{rng.randint(1, 9)}{rng.choice('ABCDEFGHJK')}{rng.randint(1, 9)}-{rng.randint(100, 999)}",
            "load_kg": round(rng.uniform(0.5, 60), 1),
            "temp_min": rng.choice([-40, -30, -20, -10, 0]),
            "temp_max": rng.choice([50, 60, 70, 85, 105]),
            "warranty_months": rng.choice([6, 12, 18, 24, 36, 48]),
            "plant": rng.choice(PLANT),
        }
        rec["text"] = (f"The {name} (part {rec['part']}) carries a maximum load of {rec['load_kg']} kg and "
                       f"runs from {rec['temp_min']} to {rec['temp_max']} °C. It ships with a "
                       f"{rec['warranty_months']}-month warranty from the {rec['plant']} plant.")
        out.append(rec)
    return out


def rag_questions(recs: list[dict]) -> list[dict]:
    """20 questions: 12 single-fact, 4 two-fact, 4 about products that are not in the catalog."""
    rng = random.Random(SEED + 1)
    picks = rng.sample(recs, 16)
    qs = []
    for r in picks[:4]:
        qs.append({"q": f"What is the maximum load of the {r['name']}?", "rec": r["id"], "expect": [f"{r['load_kg']}"]})
    for r in picks[4:8]:
        qs.append({"q": f"How long is the warranty on the {r['name']}?", "rec": r["id"],
                   "expect": [str(r["warranty_months"])]})
    for r in picks[8:12]:
        qs.append({"q": f"Which plant ships the {r['name']}?", "rec": r["id"], "expect": [r["plant"]]})
    for r in picks[12:16]:
        qs.append({"q": f"What is the part number of the {r['name']}, and what is its highest operating temperature?",
                   "rec": r["id"], "expect": [r["part"], str(r["temp_max"])]})
    names = {r["name"] for r in recs}
    while len(qs) < 20:   # a plausible name that is not in the catalog: the right answer is "not in the context"
        name = f"{rng.choice(ADJ)} {rng.randint(10, 99)} {rng.choice(NOUN)}"
        if name not in names:
            qs.append({"q": f"What is the maximum load of the {name}?", "rec": None, "expect": None})
    return qs
