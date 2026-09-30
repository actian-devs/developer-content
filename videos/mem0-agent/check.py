"""Confirm VectorAI DB is reachable before anything else."""

from vectorai_store import HOST, connect

client = connect()
print(f"connected to {HOST} | collections:", client.collections.list())
