"""Drop the agent memory collection so a run starts from nothing."""

from memory import COLLECTION
from vectorai_store import connect

client = connect()

if client.collections.exists(COLLECTION):
    client.collections.delete(COLLECTION)
    print(f"dropped {COLLECTION}")
else:
    print(f"{COLLECTION} was not there")
