"""Empty the cache so a run starts from nothing. Safe to run with the service up."""

from app.cache import COLLECTION, connect, create_collection

client = connect()

if client.collections.exists(COLLECTION):
    client.collections.delete(COLLECTION)

# recreate straight away, so a running service doesn't start 500ing
create_collection(client)
print(f"{COLLECTION} emptied | {client.points.count(COLLECTION)} points")
