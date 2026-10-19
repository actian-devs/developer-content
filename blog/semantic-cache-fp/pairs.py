"""Meaning-flip pairs for tests 2 and 3, the shared runner, and the guard.

Each pair is a cached question and an incoming question.
  same_meaning = True   a paraphrase: serving the cached answer is right
  same_meaning = False  a meaning flip: serving the cached answer is wrong

Pairs are generated from banking templates (seeded) plus hand-written ones, and
written to data/pairs_<kind>.jsonl so they can be inspected or replaced. To use
your own, pass --pairs-file with lines like:
  {"cached": "...", "incoming": "...", "same_meaning": false}
"""

import json
import random
import re

import numpy as np

from common import (
    DATA, build_collection, connect, embed, load_banking77, lookup_all, pct,
    save, summarize, THRESHOLD_GRID,
)

# ---------------------------------------------------------------- negation (test 2)
ACTIONS = [
    ("cancel", "my card"), ("freeze", "my account"), ("close", "my savings account"),
    ("renew", "my card"), ("activate", "my new card"), ("send", "the transfer"),
    ("refund", "this charge"), ("block", "contactless payments"),
    ("enable", "international payments"), ("change", "my PIN"),
    ("verify", "my identity"), ("delete", "my saved payee"),
    ("top up", "my account automatically"), ("link", "my card to Apple Pay"),
    ("share", "my statement with my partner"),
]
NEG_FLIP = [
    ("I want to {v} {o}", "I do not want to {v} {o}"),
    ("Please {v} {o}", "Please don't {v} {o}"),
    ("{V} {o} now", "Do not {v} {o}"),
    ("Should I {v} {o}?", "Should I not {v} {o}?"),
]
NEG_PARA = [
    ("I want to {v} {o}", "I'd like to {v} {o}"),
    ("Please {v} {o}", "Could you {v} {o} for me?"),
    ("{V} {o} now", "I need you to {v} {o} right away"),
    ("Should I {v} {o}?", "Is it a good idea to {v} {o}?"),
]
NEG_FLIP_HAND = [
    ("My transfer arrived", "My transfer hasn't arrived"),
    ("The ATM gave me my cash", "The ATM did not give me my cash"),
    ("My card works abroad", "My card doesn't work abroad"),
    ("I recognise this payment", "I don't recognise this payment"),
    ("My top-up was successful", "My top-up was not successful"),
    ("I received my refund", "I haven't received my refund"),
    ("My card payment went through", "My card payment did not go through"),
    ("I made this transaction", "I never made this transaction"),
    ("My new card has been delivered", "My new card has not been delivered"),
    ("The exchange rate was applied correctly", "The exchange rate wasn't applied correctly"),
    ("I can see my PIN in the app", "I can't see my PIN in the app"),
    ("My identity check passed", "My identity check did not pass"),
    ("The direct debit was taken", "The direct debit was not taken"),
    ("I authorised this payment", "I did not authorise this payment"),
    ("My account is verified", "My account is not verified"),
]
NEG_PARA_HAND = [
    ("My salary has not shown up yet", "My salary still isn't in my account"),
    ("There's a charge I don't recognise", "I do not recognise a charge on my statement"),
    ("My card was declined at the shop", "The shop declined my card"),
    ("I lost my card yesterday", "Yesterday I lost my card"),
    ("The app won't let me log in", "I can't log in to the app"),
    ("How long does a transfer take?", "How many days until a transfer arrives?"),
    ("My cash withdrawal was charged a fee", "I paid a fee on my cash withdrawal"),
    ("I was charged twice for one purchase", "One purchase was charged to me two times"),
]

# ---------------------------------------------------------------- numbers and dates (test 3)
AMOUNTS = [20, 200, 50, 500, 75, 750, 100, 1000, 150, 1500, 250, 2500]
DAYS = [3, 30, 7, 70, 14, 41, 2, 20]
DATES = ["3 March", "13 March", "30 March", "3 May", "1 June", "11 June",
         "21 July", "21 June", "5 August", "15 August", "2 October", "12 October"]
ORDINALS = ["1st", "11th", "5th", "15th", "2nd", "22nd", "3rd", "13th"]
NUM_TEMPLATES = [
    # (slot pool, cached template, flip uses same template with another value, paraphrase template)
    (AMOUNTS, "Send ${x} to my savings account", "Transfer ${x} into my savings account"),
    (AMOUNTS, "Can I withdraw ${x} today?", "Am I able to take out ${x} today?"),
    (AMOUNTS, "Set my daily card limit to ${x}", "Change my daily card limit to ${x}"),
    (DAYS, "Freeze my card for {x} days", "Lock my card for {x} days"),
    (DATES, "Schedule my rent payment for {x}", "Set my rent payment to go out on {x}"),
    (DATES, "Was I charged on {x}?", "Did I get a charge on {x}?"),
    (ORDINALS, "Move my direct debit to the {x} of the month",
     "Change my direct debit date to the {x} of the month"),
    (AMOUNTS, "Send ${x} to my savings account", "Send {x} dollars to my savings account"),
]


def _cap(s):
    return s[0].upper() + s[1:]


def generate(kind, seed=42):
    pairs = []
    if kind == "negation":
        for v, o in ACTIONS:
            for (a, b) in NEG_FLIP:
                pairs.append((a, b, False, v, o))
            for (a, b) in NEG_PARA:
                pairs.append((a, b, True, v, o))
        out = [{"cached": _cap(a.format(v=v, V=_cap(v), o=o)),
                "incoming": _cap(b.format(v=v, V=_cap(v), o=o)), "same_meaning": s}
               for a, b, s, v, o in pairs]
        out += [{"cached": a, "incoming": b, "same_meaning": False} for a, b in NEG_FLIP_HAND]
        out += [{"cached": a, "incoming": b, "same_meaning": True} for a, b in NEG_PARA_HAND]
        return out
    if kind == "numbers":
        # Cached values come from the even positions of each pool and flipped values
        # from the next position, so a flip is a near miss (3 March vs 13 March) and
        # never equals another pair's cached question.
        out = []
        for pool, tmpl, para in NUM_TEMPLATES:
            for i in range(0, len(pool) - 1, 2):
                x, y = pool[i], pool[i + 1]
                out.append({"cached": tmpl.format(x=x), "incoming": tmpl.format(x=y),
                            "same_meaning": False})
                out.append({"cached": tmpl.format(x=x), "incoming": para.format(x=x),
                            "same_meaning": True})
        return out
    raise ValueError(kind)


def load_pairs(kind, pairs_file=None):
    if pairs_file:
        with open(pairs_file) as f:
            pairs = [json.loads(line) for line in f if line.strip()]
    else:
        pairs = generate(kind)
    # Drop exact duplicates, and flips whose incoming question is word-for-word
    # another pair's cached question: serving that cached answer would be right.
    cached = {p["cached"].strip().lower() for p in pairs}
    seen, unique = set(), []
    for p in pairs:
        k = (p["cached"], p["incoming"])
        if k in seen or (not p["same_meaning"] and p["incoming"].strip().lower() in cached):
            continue
        seen.add(k)
        unique.append(p)
    path = DATA / f"pairs_{kind}.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in unique))
    n_flip = sum(not p["same_meaning"] for p in unique)
    print(f"{len(unique)} pairs ({n_flip} meaning flips, {len(unique) - n_flip} paraphrases) "
          f"-> {path}")
    return unique


# ---------------------------------------------------------------- the guard (mitigation)
NEG_RE = re.compile(r"\b(not|no|never|none|nothing|cannot|without)\b|n[’']t\b", re.I)
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
MONTHS = ("january february march april may june july august september october "
          "november december").split()
WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()


def signature(text):
    t = text.lower()
    nums = sorted(n.replace(",", "") for n in NUM_RE.findall(t))
    words = re.findall(r"[a-z]+", t)
    names = sorted(w for w in words if w in MONTHS or w in WEEKDAYS)
    return {"negated": bool(NEG_RE.search(t)), "numbers": nums, "date_words": names}


def guard(cached, incoming, kind):
    """Reject a hit when the parts similarity ignores do not match."""
    a, b = signature(cached), signature(incoming)
    if kind == "negation":
        return a["negated"] == b["negated"]
    return a["numbers"] == b["numbers"] and a["date_words"] == b["date_words"]


# ---------------------------------------------------------------- runner
def run(kind, test_no, title, stresses, args):
    pairs = load_pairs(kind, args.pairs_file)
    train, _ = load_banking77(args)

    # Meaning ids: paraphrases share their cached question's id; flips get their own.
    meaning = {}

    def mid(text, default):
        return meaning.setdefault(text.strip().lower(), default)

    cached_texts = []
    for i, p in enumerate(pairs):
        cid = mid(p["cached"], f"m{i}")
        if p["cached"] not in cached_texts:
            cached_texts.append(p["cached"])
        mid(p["incoming"], cid if p["same_meaning"] else f"m{i}_flip")
    queries = [p["incoming"] for p in pairs]
    q_mid = [meaning[q.strip().lower()] for q in queries]
    is_flip = np.array([not p["same_meaning"] for p in pairs])

    # Direct pair similarity: how close the vectors sit, before any cache.
    va = embed([p["cached"] for p in pairs])
    vb = embed(queries)
    pair_cos = (va * vb).sum(1)

    # Cache: Banking77 train as background traffic plus every cached question.
    bg_vecs = embed([r["text"] for r in train], "train")
    c_vecs = embed(cached_texts)
    vectors = np.vstack([bg_vecs, c_vecs])
    payloads = ([{"text": r["text"], "meaning": f"b77:{r['intent']}"} for r in train] +
                [{"text": t, "meaning": meaning[t.strip().lower()]} for t in cached_texts])
    client = connect()
    collection = f"fp_test{test_no}"
    build_collection(client, collection, vectors, payloads)
    scores, hits = lookup_all(client, collection, vb, label=kind)

    correct = np.array([bool(h) and h["meaning"] == m for h, m in zip(hits, q_mid)])
    s = np.nan_to_num(scores, nan=-1.0)
    all_q = np.ones(len(pairs), bool)
    kept = np.array([bool(h) and guard(h["text"], q, kind) for h, q in zip(hits, queries)])

    def at(thr, use_guard=False):
        hit = s >= thr
        if use_guard:
            hit = hit & kept
        r = summarize(hit, correct, all_q, thr)
        r["flips_served"] = float((hit & is_flip & ~correct).sum() / is_flip.sum())
        r["paraphrases_served_correctly"] = float(
            (hit & correct & ~is_flip).sum() / (~is_flip).sum())
        return r

    by_thr = [at(t) for t in THRESHOLD_GRID]
    block95 = next((r for r in by_thr if r["flips_served"] <= 0.05), None)
    before, after = at(args.threshold), at(args.threshold, use_guard=True)

    examples = sorted(
        [{"cached": h["text"], "incoming": q, "score": round(float(sc), 4)}
         for h, q, sc, f, c in zip(hits, queries, s, is_flip, correct)
         if f and h and not c and sc >= args.threshold],
        key=lambda e: -e["score"])[:15]

    print(f"\nPair cosine (median): flips {np.median(pair_cos[is_flip]):.3f}, "
          f"paraphrases {np.median(pair_cos[~is_flip]):.3f}")
    print(f"@ {args.threshold}: {pct(before['flips_served'])} of meaning flips served as a hit, "
          f"FP rate {pct(before['fp_rate'])}, paraphrases served correctly "
          f"{pct(before['paraphrases_served_correctly'])}")
    if block95:
        print(f"Blocking 95% of flips by threshold alone needs {block95['threshold']}, where "
              f"paraphrases served correctly drop to {pct(block95['paraphrases_served_correctly'])}")
    else:
        print("No threshold in the grid blocks 95% of flips")
    print(f"With the {kind} guard: flips served {pct(after['flips_served'])}, FP rate "
          f"{pct(after['fp_rate'])}, paraphrases served correctly "
          f"{pct(after['paraphrases_served_correctly'])}")
    print("\nHighest-scoring served flips:")
    for e in examples[:5]:
        print(f"  {e['score']:.3f}  '{e['incoming']}'  <-  '{e['cached']}'")

    mitigation = ("Negation check: reject a hit when one side is negated and the other is not"
                  if kind == "negation" else
                  "Exact-match check on numbers, amounts and dates before serving a hit")
    save(f"test{test_no}", {
        "test": test_no,
        "threshold": args.threshold,
        "pairs": len(pairs),
        "flip_pairs": int(is_flip.sum()),
        "pair_cosine_median": {"flips": float(np.median(pair_cos[is_flip])),
                               "paraphrases": float(np.median(pair_cos[~is_flip]))},
        "before": before,
        "after_guard": after,
        "threshold_to_block_95pct_flips": block95,
        "served_flip_examples": examples,
        "sweep": by_thr,
        "row": {
            "test": title,
            "stresses": stresses,
            "fp_before": before["fp_rate"],
            "fp_before_detail": f"{pct(before['flips_served'])} of flips served",
            "fp_after": after["fp_rate"],
            "mitigation": mitigation,
        },
    })
