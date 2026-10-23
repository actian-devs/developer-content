from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

from embed import embed_media, embed_query, load_model

COLPALI_ID = "vidore/colpali-v1.3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="pages.json")
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    pages = json.loads(Path(args.pages).read_text())
    queries = [q for q in json.loads(Path(args.queries).read_text())
               if q["query_modality"] == "text" and q["target_modality"] == "manual_page"]

    if torch.cuda.is_available():
        device, dtype = "cuda:0", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32   # works, but slow for a 3B model
    cp = ColPali.from_pretrained(COLPALI_ID, torch_dtype=dtype, device_map=device).eval()
    proc = ColPaliProcessor.from_pretrained(COLPALI_ID)

    t0 = time.perf_counter()
    page_embs = []
    for i in range(0, len(pages), 4):
        batch = proc.process_images([Image.open(p["path"]).convert("RGB") for p in pages[i : i + 4]]).to(cp.device)
        with torch.no_grad():
            page_embs.extend(list(cp(**batch).cpu()))
    cp_index_s = time.perf_counter() - t0
    bytes_per_page = page_embs[0].numel() * page_embs[0].element_size()

    gemma = load_model()
    t0 = time.perf_counter()
    gem_page = np.stack([embed_media(gemma, "image", p["path"]) for p in pages])
    gm_index_s = time.perf_counter() - t0
    gem_bytes = gem_page.shape[1] * 2  # bf16

    hits_cp = hits_gm = 0
    for q in queries:
        qb = proc.process_queries([q["query"]]).to(cp.device)
        with torch.no_grad():
            qe = cp(**qb)
        scores = proc.score_multi_vector(qe, page_embs)[0]
        top_cp = scores.topk(min(args.k, len(pages))).indices.tolist()
        top_gm = np.argsort(-(gem_page @ embed_query(gemma, q["query"])))[: args.k].tolist()
        hits_cp += any(pages[i]["asset_id"] in q["relevant"] for i in top_cp)
        hits_gm += any(pages[i]["asset_id"] in q["relevant"] for i in top_gm)

    n = len(queries)
    print(f"queries: {n}")
    print(f"EmbeddingGemma 2  hit@{args.k} {hits_gm / n:.3f}   {gem_bytes} bytes/page (bf16, one vector)   indexed in {gm_index_s:.1f}s")
    print(f"ColPali           hit@{args.k} {hits_cp / n:.3f}   {bytes_per_page} bytes/page "
          f"({page_embs[0].shape[0]} vectors x {page_embs[0].shape[1]})   indexed in {cp_index_s:.1f}s")


if __name__ == "__main__":
    main()