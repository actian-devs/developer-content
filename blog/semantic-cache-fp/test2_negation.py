"""Test 2: paraphrase against negation.

What it stresses
  "Cancel my card" and "do not cancel my card" sit close in embedding space but
  need opposite answers. The test caches one side of each pair and looks up the
  other, with Banking77 train in the cache as background traffic.

  Equal-sized controls: paraphrase pairs that should hit. A threshold high
  enough to block the flips also blocks the paraphrases; the test measures both.

Mitigation measured: reject a hit when one side is negated and the other is not.
"""

from common import base_args
from pairs import run


def main():
    p = base_args(__doc__.splitlines()[0])
    p.add_argument("--pairs-file", help="JSONL of your own pairs (see pairs.py)")
    args = p.parse_args()
    run("negation", 2, "2. Paraphrase against negation",
        "Negated questions that score close to the cached original", args)


if __name__ == "__main__":
    main()
