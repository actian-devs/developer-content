import json
import threading

import numpy as np
import pytest
from actian_vectorai import CapacityExceededError, Distance, VectorAITimeoutError

from chroma2actian.errors import CapacityError, Interrupted, MigrationError, UsageError
from chroma2actian.pipeline import MigrationOptions, migrate_collection
from chroma2actian.transform import IdStrategy, TransformOptions, canonical_id, map_id
from chroma2actian.validate import ValidateOptions, run_validation


def mopts(tmp_path, topts, name="docs", **kw):
    kw.setdefault("workdir", tmp_path / "wd")
    kw.setdefault("batch_size", 40)
    kw.setdefault("progress_interval", 1e9)
    return MigrationOptions(collection=name, target_collection=kw.pop("target", name), transform=topts, **kw)


def vopts(tmp_path, topts, name="docs", **kw):
    return ValidateOptions(collection=name, target_collection=name, transform=topts, workdir=tmp_path / "wd",
                           poll_timeout=0.1, **kw)


@pytest.mark.parametrize("space,dist", [("l2", Distance.Euclid), ("cosine", Distance.Cosine), ("ip", Distance.Dot)])
def test_migrate_and_validate_each_metric(tmp_path, make_collection, source, target, fake, topts, space, dist):
    make_collection(space=space)
    s = migrate_collection(source, target, mopts(tmp_path, topts))
    assert s["status"] == "ok" and s["migrated"] == 237 and s["distance"] == dist.name
    coll = fake.data["docs"]
    assert coll.distance == dist and len(coll.rows) == 237
    pid = map_id("doc-5")
    assert coll.rows[pid][1] == {"idx": 5, "grp": "g0", "score": 5 / 3.0, "flag": False, "chroma_id": "doc-5", "document": "text 5"}
    assert fake.vde.flushed == ["docs"]
    rep = run_validation(source, target, vopts(tmp_path, topts))
    assert rep["ok"], rep
    assert {c["name"] for c in rep["checks"]} >= {"config", "target_count", "id_set", "sample_records", "search_parity"}


def test_numeric_and_preserve_id_strategies(tmp_path, make_collection, source, target, fake):
    make_collection(ids=[str(i) for i in range(50)], n=50)
    t = TransformOptions(id_strategy=IdStrategy.NUMERIC)
    migrate_collection(source, target, mopts(tmp_path, t))
    assert set(fake.data["docs"].rows) == set(range(50))
    assert run_validation(source, target, vopts(tmp_path, t))["ok"]


def test_resume_after_interrupt_has_no_gaps_or_duplicates(tmp_path, make_collection, source, target, fake, topts):
    make_collection(n=300)
    stop = threading.Event()
    calls = {"n": 0}

    def hook(points):
        calls["n"] += 1
        if calls["n"] == 8:
            stop.set()

    fake.on_upsert = hook
    with pytest.raises(Interrupted):
        migrate_collection(source, target, mopts(tmp_path, topts, stop_event=stop, workers=1))
    state = json.loads((tmp_path / "wd" / "state.json").read_text())
    assert 0 < state["next_offset"] < 300 and not state["completed"]
    partial = len(fake.data["docs"].rows)
    assert 0 < partial < 300

    with pytest.raises(UsageError, match="--resume"):
        migrate_collection(source, target, mopts(tmp_path, topts))
    fake.on_upsert = None
    s = migrate_collection(source, target, mopts(tmp_path, topts, resume=True))
    assert s["status"] == "ok" and len(fake.data["docs"].rows) == 300
    assert run_validation(source, target, vopts(tmp_path, topts))["ok"]


def test_completed_run_needs_fresh_and_settings_change_is_detected(tmp_path, make_collection, source, target, topts):
    make_collection(n=60)
    migrate_collection(source, target, mopts(tmp_path, topts))
    with pytest.raises(UsageError, match="already completed"):
        migrate_collection(source, target, mopts(tmp_path, topts))
    changed = TransformOptions(document_field="body")
    with pytest.raises(UsageError, match="different migration"):
        migrate_collection(source, target, mopts(tmp_path, changed, resume=True))
    migrate_collection(source, target, mopts(tmp_path, topts, fresh=True, if_exists="reuse"))


def test_if_exists_policies(tmp_path, make_collection, source, target, fake, topts):
    make_collection(n=60)
    migrate_collection(source, target, mopts(tmp_path, topts))
    with pytest.raises(UsageError, match="already exists"):
        migrate_collection(source, target, mopts(tmp_path, topts, fresh=True))
    with pytest.raises(UsageError, match="--yes"):
        migrate_collection(source, target, mopts(tmp_path, topts, fresh=True, if_exists="recreate"))
    s = migrate_collection(source, target, mopts(tmp_path, topts, fresh=True, if_exists="recreate", assume_yes=True))
    assert s["status"] == "ok" and len(fake.data["docs"].rows) == 60


def test_reuse_with_extra_points_validates_in_subset_mode(tmp_path, make_collection, source, target, fake, topts):
    make_collection(n=60)
    target.create("docs", 16, Distance.Cosine)
    from actian_vectorai import PointStruct
    target.upsert("docs", [PointStruct(id=999999, vector=[1.0] * 16, payload={"x": 1})])
    s = migrate_collection(source, target, mopts(tmp_path, topts, if_exists="reuse"))
    assert s["preexisting_points"] == 1 and s["target_count"] == 61
    assert run_validation(source, target, vopts(tmp_path, topts))["ok"]


def test_existing_collection_with_wrong_shape_is_refused(tmp_path, make_collection, source, target, topts):
    make_collection(n=20)
    target.create("docs", 8, Distance.Cosine)
    with pytest.raises(UsageError, match="dimension=8"):
        migrate_collection(source, target, mopts(tmp_path, topts, if_exists="reuse"))


def test_source_changing_during_migration_fails(tmp_path, make_collection, source, target, topts, chroma_path):
    make_collection(n=120)
    real = source.count
    n = {"i": 0}

    def flaky(coll):
        n["i"] += 1
        return real(coll) + (1 if n["i"] > 1 else 0)

    source.count = flaky
    with pytest.raises(MigrationError, match="count changed"):
        migrate_collection(source, target, mopts(tmp_path, topts))


def test_bad_records_fail_by_default_and_skip_when_asked(tmp_path, make_collection, source, target, fake, topts):
    vecs = np.random.default_rng(1).normal(size=(40, 8)).astype(np.float32)
    vecs[7] = 0.0
    make_collection(n=40, dim=8, vectors=vecs, space="cosine")
    with pytest.raises(MigrationError, match="zero vector"):
        migrate_collection(source, target, mopts(tmp_path, topts))
    s = migrate_collection(source, target, mopts(tmp_path, topts, fresh=True, if_exists="recreate", assume_yes=True,
                                                 on_bad_record="skip"))
    assert s["migrated"] == 39 and s["skipped"] == 1
    rej = [json.loads(l) for l in (tmp_path / "wd" / "rejected.jsonl").read_text().splitlines()]
    assert rej[0]["chroma_id"] == "doc-7"
    assert run_validation(source, target, vopts(tmp_path, topts))["ok"]
    # --allow-zero-vectors keeps it
    t2 = TransformOptions()
    s2 = migrate_collection(source, target, mopts(tmp_path, t2, fresh=True, if_exists="recreate", assume_yes=True,
                                                  allow_zero_vectors=True))
    assert s2["migrated"] == 40


def test_message_size_errors_split_batches(tmp_path, make_collection, source, target, fake, topts):
    make_collection(n=200)
    fake.max_message_points = 7
    migrate_collection(source, target, mopts(tmp_path, topts, batch_size=64))
    assert len(fake.data["docs"].rows) == 200


def test_transient_errors_are_retried(tmp_path, make_collection, source, target, fake, topts):
    make_collection(n=80)
    seen = {"n": 0}

    def hook(points):
        seen["n"] += 1
        if seen["n"] in (1, 2):
            raise VectorAITimeoutError("slow")

    fake.on_upsert = hook
    migrate_collection(source, target, mopts(tmp_path, topts, workers=1))
    assert len(fake.data["docs"].rows) == 80


def test_capacity_exhaustion_is_fatal_and_clear(tmp_path, make_collection, source, target, fake, topts):
    make_collection(n=100)
    fake.max_points = 30
    with pytest.raises(CapacityError, match="licence"):
        migrate_collection(source, target, mopts(tmp_path, topts, workers=1))


def test_empty_source_is_not_an_error(tmp_path, make_collection, source, target, topts, chroma_path):
    import chromadb
    chromadb.PersistentClient(path=str(chroma_path)).create_collection("docs")
    s = migrate_collection(source, target, mopts(tmp_path, topts))
    assert s["status"] == "empty"


# --------------------------------------------------------- validation catches damage
def _migrated(tmp_path, make_collection, source, target, topts):
    make_collection(n=150)
    migrate_collection(source, target, mopts(tmp_path, topts))


def test_validation_detects_missing_point(tmp_path, make_collection, source, target, fake, topts):
    _migrated(tmp_path, make_collection, source, target, topts)
    gone = map_id("doc-33")
    del fake.data["docs"].rows[gone]
    rep = run_validation(source, target, vopts(tmp_path, topts))
    by = {c["name"]: c for c in rep["checks"]}
    assert not rep["ok"] and not by["id_set"]["ok"] and not by["target_count"]["ok"]
    assert by["id_set"]["missing_in_target"] == 1 and by["id_set"]["missing_examples"] == [canonical_id(gone)]


def test_validation_detects_swapped_point_with_same_count(tmp_path, make_collection, source, target, fake, topts):
    _migrated(tmp_path, make_collection, source, target, topts)
    rows = fake.data["docs"].rows
    rows[map_id("rogue")] = rows.pop(map_id("doc-1"))
    by = {c["name"]: c for c in run_validation(source, target, vopts(tmp_path, topts))["checks"]}
    assert by["target_count"]["ok"] and not by["id_set"]["ok"]
    assert by["id_set"]["unexpected_in_target"] == 1


def test_validation_detects_corrupt_payload_and_vector(tmp_path, make_collection, source, target, fake, topts):
    _migrated(tmp_path, make_collection, source, target, topts)
    rows = fake.data["docs"].rows
    for cid in (f"doc-{i}" for i in range(150)):          # corrupt everything so the hash-sample must hit it
        v, p = rows[map_id(cid)]
        rows[map_id(cid)] = (v[::-1].copy(), {**p, "grp": "tampered"})
    by = {c["name"]: c for c in run_validation(source, target, vopts(tmp_path, topts, sample=100))["checks"]}
    assert not by["sample_records"]["ok"]
    problems = {m["problem"] for m in by["sample_records"]["examples"]}
    assert "payload differs" in problems and "vector differs" in problems


def test_validation_detects_wrong_metric(tmp_path, make_collection, source, target, fake, topts):
    _migrated(tmp_path, make_collection, source, target, topts)
    fake.data["docs"].distance = Distance.Euclid
    by = {c["name"]: c for c in run_validation(source, target, vopts(tmp_path, topts))["checks"]}
    assert not by["config"]["ok"]


def test_cosine_normalisation_by_server_is_not_a_false_positive(tmp_path, make_collection, source, target, fake, topts):
    assert fake.normalize_cosine
    _migrated(tmp_path, make_collection, source, target, topts)
    stored = next(iter(fake.data["docs"].rows.values()))[0]
    assert abs(np.linalg.norm(stored) - 1.0) < 1e-5            # the fake really did normalise
    assert run_validation(source, target, vopts(tmp_path, topts))["ok"]


def test_golden_queries_file(tmp_path, make_collection, source, target, topts):
    _migrated(tmp_path, make_collection, source, target, topts)
    rng = np.random.default_rng(3)
    g = tmp_path / "golden.jsonl"
    g.write_text("\n".join(json.dumps({"query_id": f"q{i}", "embedding": rng.normal(size=16).tolist()}) for i in range(5)))
    rep = run_validation(source, target, vopts(tmp_path, topts, golden_queries=g))
    sp = next(c for c in rep["checks"] if c["name"] == "search_parity")
    assert sp["ok"] and sp["query_kind"] == "golden" and sp["queries"] == 5
