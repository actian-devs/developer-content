# smoke_test.py
from importlib import import_module

VectorAIClient = import_module("actian_vectorai").VectorAIClient

with VectorAIClient("localhost:6574") as client:
    print(client.health_check())