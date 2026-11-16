"""Opt-in smoke test against a REAL Actian VectorAI DB (docker compose up -d).

    ACTIAN_LIVE=1 pytest tests/test_live_actian.py            # optional: ACTIAN_VECTORAI_URL=host:6574
"""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("ACTIAN_LIVE") != "1", reason="set ACTIAN_LIVE=1 (needs a running server)")


def test_live_round_trip(tmp_path, make_collection, source, topts):
    from chroma2actian.config import ActianConfig
    from chroma2actian.pipeline import MigrationOptions, migrate_collection
    from chroma2actian.target import ActianTarget
    from chroma2actian.validate import ValidateOptions, run_validation

    make_collection("c2a_live", n=500, dim=32, space="cosine")
    target = ActianTarget(ActianConfig(wait_for_target=30))
    target.connect()
    try:
        name = "c2a_live_target"
        if target.exists(name):
            target.drop(name)
        migrate_collection(source, target, MigrationOptions(collection="c2a_live", target_collection=name, transform=topts,
                                                            workdir=tmp_path / "wd", progress_interval=1e9))
        rep = run_validation(source, target, ValidateOptions(collection="c2a_live", target_collection=name,
                                                             transform=topts, workdir=tmp_path / "wd"))
        assert rep["ok"], rep
    finally:
        if target.exists("c2a_live_target"):
            target.drop("c2a_live_target")
        target.close()
