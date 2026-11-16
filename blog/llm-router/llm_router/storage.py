"""Durable attempt journal plus the existing Actian tutorial collection only."""
import json
import math
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager
from .contract import assert_runtime_only, canonical, fingerprint

COLLECTION = 'vectorai_llm_router_tutorial_attempts_v1'


def vector_check(vector):
    if len(vector) != 768 or not all(math.isfinite(v) for v in vector) or not any(vector):
        raise ValueError('Expected a nonzero finite 768-dimensional EmbeddingGemma vector')


class Journal:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, record TEXT NOT NULL)')

    @contextmanager
    def db(self):
        db=sqlite3.connect(self.path,timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, record):
        assert_runtime_only(record)
        with self.db() as db:
            db.execute('PRAGMA synchronous=FULL')
            existing = db.execute('SELECT record FROM attempts WHERE id=?', (record['attempt_id'],)).fetchone()
            if existing and json.loads(existing[0]).get('status') not in ['started']:
                raise ValueError('Completed attempt IDs are immutable')
            db.execute('INSERT OR REPLACE INTO attempts VALUES (?,?)', (record['attempt_id'], canonical(record)))

    def all(self):
        with self.db() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT record FROM attempts ORDER BY rowid')]


class ActianStore:
    def __init__(self, endpoint='localhost:6674'):
        if endpoint != 'localhost:6674':
            raise ValueError('This project is bound to the selected existing tutorial endpoint')
        from actian_vectorai import VectorAIClient
        self.client = VectorAIClient(endpoint)
        self.client.connect()
        info = self.client.collections.get_info(COLLECTION).model_dump(mode='json')
        self.collection_info = info
        # Confirm vector configuration using the actual SDK model representation.
        cfg = info.get('config', {}).get('params', {}).get('vectors', {})
        if not cfg:
            cfg = info.get('params', {}).get('vectors', {})
        self.endpoint = endpoint

    def write(self, record, vector):
        from actian_vectorai import PointStruct
        assert_runtime_only(record); vector_check(vector)
        # Unique attempt UUIDs; never update another attempt or any preexisting project point.
        if self.read([record['attempt_id']]):
            raise ValueError('Actian attempt ID already exists; refusing overwrite')
        self.client.points.upsert(COLLECTION, [PointStruct(id=record['attempt_id'], vector=vector, payload=record)], timeout=15)
        for _ in range(20):
            got = self.read([record['attempt_id']])
            if got and fingerprint(got[0]['payload']) == fingerprint(record):
                return
            time.sleep(.1)
        raise RuntimeError('Actian write not verified; stop rather than continue with incomplete memory')

    def read(self, ids):
        if not ids: return []
        return [r.model_dump(mode='json') for r in self.client.points.get(COLLECTION, ids, with_payload=True, timeout=15)]

    def search(self, vector, eligible_ids):
        from actian_vectorai import FilterBuilder, has_id
        vector_check(vector)
        if not eligible_ids: return []
        # Search all frozen eligible attempts, then deduplicate. Top-10 attempts would
        # incorrectly hide distinct messages behind repeated successes/failures.
        filt = FilterBuilder().must(has_id(list(eligible_ids))).build()
        rows = self.client.points.search(COLLECTION, vector, filter=filt, limit=len(eligible_ids), with_payload=True, timeout=15)
        return [r.model_dump(mode='json') for r in rows]

    def delete_test_ids(self, ids, test_namespace):
        rows = self.read(ids)
        if any(r['payload'].get('namespace') != test_namespace or r['payload'].get('stage') != 'verification' for r in rows):
            raise ValueError('Refusing cleanup of a record not owned by this verification')
        present = [r['id'] for r in rows]
        if present:
            self.client.points.delete(COLLECTION, ids=present, strict=True, timeout=15)

    def close(self):
        self.client.close()
