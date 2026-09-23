# benchmark.py
def measure_recall_at_k(client, encoder, collection, test_records, k=5) -> float:
    hits = 0
    for record in test_records:
        vec = encoder.encode_images([record["image_path"]])[0].tolist()
        results = client.points.search(
            collection, vector=vec, limit=k + 1, with_payload=True
        )
        neighbours = [
            r for r in results if r.payload["image_path"] != record["image_path"]
        ][:k]
        if any(r.payload["defect_type"] == record["defect_type"] for r in neighbours):
            hits += 1
    return hits / len(test_records) if test_records else 0.0