"""Test 3: numbers against dates.

What it stresses
  "Send $50 to savings" and "Send $500 to savings", or a rent payment on
  3 March against 13 March: one token changes the answer, and the vectors
  barely move. The test caches one side of each pair and looks up the other,
  with Banking77 train in the cache as background traffic.

  Equal-sized controls: paraphrases that keep the same values, including
  "$50" against "50 dollars".

Mitigation measured: exact match on numbers, amounts and date words before a hit is served.
"""

from common import base_args
from pairs import run


def main():
    p = base_args(__doc__.splitlines()[0])
    p.add_argument("--pairs-file", help="JSONL of your own pairs (see pairs.py)")
    args = p.parse_args()
    run("numbers", 3, "3. Numbers against dates",
        "Questions that differ only in an amount, a count or a date", args)


if __name__ == "__main__":
    main()
