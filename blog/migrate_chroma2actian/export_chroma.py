#!/usr/bin/env python3
"""Export Chroma collections (ids, embeddings, documents, metadatas) to one JSONL file each.

  python export_chroma.py --source local --chroma-path ./chroma_demo --all-collections
  python export_chroma.py --source http  --chroma-host localhost --chroma-port 8000 --collection my_docs
  CHROMA_API_KEY=.. CHROMA_TENANT=.. CHROMA_DATABASE=.. python export_chroma.py --source cloud --all-collections

Pause writes to Chroma first: offset paging is not a consistent snapshot on a changing collection.
"""
import argparse, json, logging, os, sys
import chromadb

log = logging.getLogger("export")


def connect(a):
    if a.source == "local":
        return chromadb.PersistentClient(path=a.chroma_path)
    if a.source == "http":
        headers = {"Authorization": f"Bearer {a.chroma_token}"} if a.chroma_token else None
        return chromadb.HttpClient(host=a.chroma_host, port=a.chroma_port, ssl=a.chroma_ssl, headers=headers)
    if not (a.chroma_api_key and a.chroma_tenant and a.chroma_database):
        sys.exit("cloud needs CHROMA_API_KEY, CHROMA_TENANT and CHROMA_DATABASE (or the --chroma-* flags)")
    return chromadb.CloudClient(tenant=a.chroma_tenant, database=a.chroma_database, api_key=a.chroma_api_key)


def export_collection(col, out_dir, page_size):
    os.makedirs(out_dir, exist_ok=True)
    final, tmp = f"{out_dir}/{col.name}.jsonl", f"{out_dir}/{col.name}.jsonl.tmp"
    expected, offset, written = col.count(), 0, 0
    with open(tmp, "w", encoding="utf-8") as f:
        while True:
            page = col.get(limit=page_size, offset=offset, include=["embeddings", "documents", "metadatas"])
            ids = page["ids"]
            if not ids:
                break
            for i, _id in enumerate(ids):
                f.write(json.dumps({"id": _id, "embedding": [float(x) for x in page["embeddings"][i]],
                                    "document": page["documents"][i], "metadata": page["metadatas"][i]}) + "\n")
            log.info("%s: offset=%d page=%d total=%d/%d", col.name, offset, len(ids), written + len(ids), expected)
            offset += len(ids)
            written += len(ids)
        f.flush()
        os.fsync(f.fileno())
    if written != expected or col.count() != expected:
        raise RuntimeError(f"{col.name}: wrote {written}, count() was {expected} then {col.count()} "
                           "(was the collection being written to?). Partial file left as " + tmp)
    os.replace(tmp, final)  # real name only once complete and verified
    return final, written


def main():
    p = argparse.ArgumentParser(description="Export Chroma collections to JSONL")
    p.add_argument("--source", required=True, choices=("local", "http", "cloud"))
    p.add_argument("--chroma-path", default=os.environ.get("CHROMA_PATH"))
    p.add_argument("--chroma-host", default=os.environ.get("CHROMA_HOST", "localhost"))
    p.add_argument("--chroma-port", type=int, default=int(os.environ.get("CHROMA_PORT", "8000")))
    p.add_argument("--chroma-ssl", action="store_true")
    p.add_argument("--chroma-token", default=os.environ.get("CHROMA_TOKEN"))
    p.add_argument("--chroma-api-key", default=os.environ.get("CHROMA_API_KEY"))
    p.add_argument("--chroma-tenant", default=os.environ.get("CHROMA_TENANT"))
    p.add_argument("--chroma-database", default=os.environ.get("CHROMA_DATABASE"))
    p.add_argument("--collection", action="append", default=[])
    p.add_argument("--all-collections", action="store_true")
    p.add_argument("--out-dir", default="backup")
    p.add_argument("--page-size", type=int, default=0, help="default 1000 local/http, 100 cloud")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if a.source == "local" and not a.chroma_path:
        sys.exit("--source local needs --chroma-path")

    client = connect(a)
    names = [c if isinstance(c, str) else c.name for c in client.list_collections()] if a.all_collections else a.collection
    if not names:
        sys.exit("pass --collection NAME (repeatable) or --all-collections")
    page_size = a.page_size or (100 if a.source == "cloud" else 1000)
    for n in names:
        path, rows = export_collection(client.get_collection(n), a.out_dir, page_size)
        print(f"exported {rows} records -> {path}")


if __name__ == "__main__":
    main()
