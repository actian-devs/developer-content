"""Check the database and the model before indexing anything."""
from __future__ import annotations

import sys

import numpy as np
from actian_vectorai import VectorAIClient
from PIL import Image

from embed import embed_document, embed_media, embed_query, load_model

with VectorAIClient("localhost:6584") as client:
    print("VectorAI DB:", client.health_check())

model = load_model()
print("model dtype:", next(model.parameters()).dtype)

q = embed_query(model, "replace the impeller seal")
d = embed_document(model, "To replace the impeller seal, drain the pump housing first.")
img_path = sys.argv[1] if len(sys.argv) > 1 else "smoke.png"
if len(sys.argv) == 1:
    Image.new("RGB", (224, 224), "white").save(img_path)
i = embed_media(model, "image", img_path)

for name, v in [("query", q), ("document", d), ("image", i)]:
    print(f"{name:<9} shape {v.shape}  norm {np.linalg.norm(v):.3f}  finite {np.isfinite(v).all()}")
print(f"query-document cosine {float(q @ d):.3f}")