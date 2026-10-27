"""Development/test split, the threshold-only baseline, and the three bands.

Everything is tuned on the development split and reported on the test split, so the band
edges and the best cutoff never see the records they are scored on.
"""
import random

import numpy as np


def split(left_ids, dev_share: float, seed: int = 42) -> tuple[set, set]:
    """Split by left record, so all of a record's candidate pairs land on the same side."""
    ids = sorted(left_ids)
    random.Random(seed).shuffle(ids)
    cut = int(len(ids) * dev_share)
    return set(ids[:cut]), set(ids[cut:])


def subset(candidates: dict, gold: set, left_ids: set):
    """Candidates and gold pairs whose left record is in `left_ids`."""
    cand = {p: s for p, s in candidates.items() if p[0] in left_ids}
    return cand, {p for p in gold if p[0] in left_ids}


def prf(predicted: set, gold: set) -> dict:
    """Precision, recall, F1. Recall counts true matches lost at blocking as misses."""
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "true_positives": tp,
            "predicted": len(predicted), "gold": len(gold)}


def best_threshold(cand: dict, gold: set) -> float:
    """The single cutoff that maximises F1 (the threshold-only baseline)."""
    pairs = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)
    s = np.array([score for _, score in pairs])
    tp = np.cumsum([p in gold for p, _ in pairs])
    f1 = 2 * tp / (np.arange(1, len(s) + 1) + len(gold))   # F1 = 2TP / (predicted + gold)
    last_of_tie = np.r_[s[1:] != s[:-1], True]              # a cutoff can't split tied scores
    return float(s[np.argmax(np.where(last_of_tie, f1, -1))])


def band_edges(cand: dict, gold: set, target_precision: float, max_miss: float) -> tuple[float, float]:
    """Upper edge: lowest score above which predicted matches reach `target_precision`.
    Lower edge: highest score below which at most `max_miss` of the candidate matches fall."""
    pairs = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)
    s = np.array([score for _, score in pairs])
    y = np.array([p in gold for p, _ in pairs], dtype=float)
    if y.sum() == 0:
        raise ValueError("no true matches among development candidates")

    precision_above = np.cumsum(y) / np.arange(1, len(y) + 1)
    ok = np.where(precision_above >= target_precision)[0]
    upper = s[ok.max()] if len(ok) else s[0] + 1e-9    # nothing reaches the target: auto-match nothing

    missed_below = (y.sum() - np.cumsum(y)) / y.sum()   # share of matches scoring below each position
    lower = s[np.where(missed_below <= max_miss)[0].min()]
    return float(min(lower, upper)), float(upper)


def rank_and_gap(candidates: dict) -> dict:
    """{pair: (rank, gap)}: the pair's rank among its left record's candidates (1 = nearest)
    and how far its score sits below that record's best candidate."""
    by_left = {}
    for (left, right), score in candidates.items():
        by_left.setdefault(left, []).append((score, right))
    info = {}
    for left, items in by_left.items():
        items.sort(reverse=True)
        best = items[0][0]
        for rank, (score, right) in enumerate(items, 1):
            info[(left, right)] = (rank, best - score)
    return info


def best_match(cand: dict, info: dict) -> dict:
    """Only each left record's nearest candidate (rank 1)."""
    return {p: s for p, s in cand.items() if info[p][0] == 1}


def near_tie_band(cand: dict, info: dict, lower: float, upper: float, max_rank: int, max_gap: float) -> list:
    """The pairs sent to Argon: rank-1 pairs scoring between the edges, plus lower-ranked
    candidates (rank 2..max_rank) scoring within `max_gap` of their record's best, above the lower edge."""
    band = []
    for p, s in cand.items():
        rank, gap = info[p]
        if s < lower:
            continue
        if (rank == 1 and s < upper) or (2 <= rank <= max_rank and gap <= max_gap):
            band.append(p)
    return sorted(band)


def widen(lower: float, upper: float, scale: float) -> tuple[float, float]:
    """Scale the band's width around its centre (scale 2.0 doubles it), for the band-width experiment."""
    centre, half = (lower + upper) / 2, (upper - lower) / 2 * scale
    return centre - half, centre + half
