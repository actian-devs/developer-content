"""List every collection in VectorAI DB with its vector count, and optionally delete.

  uv run python cleanup.py                 # list only
  uv run python cleanup.py --tests         # delete the fp_test1..fp_test7 collections
  uv run python cleanup.py --name NAME     # delete one collection by name
"""

import argparse

from common import TEST_COLLECTIONS, connect


def names(client):
    return [getattr(c, "name", c) for c in (client.collections.list() or [])]


def show(client, title):
    print(f"\n{title}")
    total = 0
    for n in names(client):
        try:
            count = client.points.count(n, exact=True)
        except Exception as e:
            count = f"? ({type(e).__name__})"
        total += count if isinstance(count, int) else 0
        print(f"  {n:<30} {count}")
    print(f"  {'TOTAL':<30} {total}  (Community Edition limit: 5,000)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tests", action="store_true", help="Delete fp_test1..fp_test7")
    p.add_argument("--name", action="append", default=[], help="Delete this collection")
    args = p.parse_args()

    client = connect()
    show(client, "Collections now:")

    targets = (TEST_COLLECTIONS if args.tests else []) + args.name
    if not targets:
        return
    existing = set(names(client))
    for n in targets:
        if n in existing:
            client.collections.delete(n)
            print(f"Deleted '{n}'")
    show(client, "Collections after delete:")


if __name__ == "__main__":
    main()
