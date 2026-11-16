import json

import chromadb
import pytest
from chromadb import errors as ce

from chroma2actian import cli
from chroma2actian.config import ChromaConfig
from chroma2actian.errors import SourceError, UsageError
from chroma2actian.source import ChromaSource
from chroma2actian.transform import map_id


# ------------------------------------------------------------------ config
def test_cloud_config_requires_credentials_and_masks_secrets():
    with pytest.raises(UsageError, match="CHROMA_API_KEY"):
        ChromaConfig(mode="cloud").validate()
    c = ChromaConfig(mode="cloud", api_key="sk-supersecret-1234", tenant="t", database="d")
    c.validate()
    assert "supersecret" not in json.dumps(c.safe_dict()) and "supersecret" not in c.describe()
    assert c.page_size == 100 and ChromaConfig(mode="http").page_size == 1000


def test_local_config_requires_existing_dir(tmp_path):
    with pytest.raises(UsageError):
        ChromaConfig(mode="local", path=str(tmp_path / "nope")).validate()


def test_connect_builds_the_right_chroma_client(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(chromadb, "CloudClient", lambda **k: seen.setdefault("cloud", k) or object())
    monkeypatch.setattr(chromadb, "HttpClient", lambda **k: seen.setdefault("http", k) or object())
    ChromaSource(ChromaConfig(mode="cloud", api_key="k", tenant="t", database="d")).connect()
    assert seen["cloud"] == {"tenant": "t", "database": "d", "api_key": "k", "cloud_host": "api.trychroma.com"}
    ChromaSource(ChromaConfig(mode="http", host="h", port=9, ssl=True, token="tok", tenant="t", database="d")).connect()
    assert seen["http"]["headers"] == {"Authorization": "Bearer tok"} and seen["http"]["host"] == "h" and seen["http"]["ssl"]


# ------------------------------------------------------------------ source
class FlakyColl:
    """Collection whose get() rejects page sizes > 50 (like a server-side limit) or fails with a given error."""

    def __init__(self, n, exc=None, max_limit=50):
        self.n, self.exc, self.max_limit, self.calls = n, exc, max_limit, 0

    def get(self, include, limit, offset, **_):
        self.calls += 1
        if self.exc and not isinstance(self.exc, ce.QuotaError):
            raise self.exc
        if limit > self.max_limit:
            raise ce.QuotaError("limit too large")
        ids = [f"i{i}" for i in range(offset, min(offset + limit, self.n))]
        return {"ids": ids, "embeddings": [[1.0]] * len(ids), "documents": None, "metadatas": None}


def test_page_size_shrinks_on_quota_errors():
    s = ChromaSource(ChromaConfig(mode="local", default_page_size=400), client=object(), sleep=lambda _s: None)
    got = [i for p in s.iter_pages(FlakyColl(180)) for i in p.ids]
    assert got == [f"i{i}" for i in range(180)] and s.page_size <= 50


def test_fatal_errors_are_not_retried():
    s = ChromaSource(ChromaConfig(mode="local"), client=object(), sleep=lambda _s: None)
    coll = FlakyColl(10, exc=ce.ChromaAuthError("bad key"))
    with pytest.raises(SourceError, match="bad key"):
        list(s.iter_pages(coll))
    assert coll.calls == 1


def test_space_detection(chroma_path):
    c = chromadb.PersistentClient(path=str(chroma_path))
    c.create_collection("aaa", configuration={"hnsw": {"space": "ip"}})
    c.create_collection("bbb")
    s = ChromaSource(ChromaConfig(mode="local", path=str(chroma_path)))
    assert s.space(s.get_collection("aaa")) == ("ip", "configuration")
    assert s.space(s.get_collection("bbb"))[0] == "l2"

    class Legacy:
        configuration = None
        metadata = {"hnsw:space": "cosine"}

    assert s.space(Legacy()) == ("cosine", "metadata")

    class Bare:
        configuration = None
        metadata = None

    assert s.space(Bare()) == ("l2", "default")


def test_list_collections_and_missing_collection(chroma_path, make_collection):
    make_collection("one", n=3)
    make_collection("two", n=3)
    s = ChromaSource(ChromaConfig(mode="local", path=str(chroma_path)))
    assert sorted(s.list_collections()) == ["one", "two"]
    with pytest.raises(UsageError, match="nope"):
        s.get_collection("nope")


# --------------------------------------------------------------------- CLI
@pytest.fixture
def wired(monkeypatch, fake, tmp_path):
    from chroma2actian.target import ActianTarget
    monkeypatch.setattr(cli, "target_factory", lambda cfg: ActianTarget(cfg, client=fake, sleep=lambda _s: None))
    return fake


def base(chroma_path, tmp_path, *extra):
    return ["migrate", "--source", "local", "--chroma-path", str(chroma_path), "--workdir", str(tmp_path / "wd"),
            "--poll-timeout", "0.1", *extra]


def test_cli_end_to_end_multiple_collections_with_prefix(wired, make_collection, chroma_path, tmp_path):
    make_collection("aaa", n=30)
    make_collection("bbb", n=45, space="l2")
    assert cli.run(base(chroma_path, tmp_path, "--all-collections", "--target-prefix", "v2_")) == 0
    assert {k: len(v.rows) for k, v in wired.data.items()} == {"v2_aaa": 30, "v2_bbb": 45}
    assert (tmp_path / "wd" / "aaa" / "validation_report.json").exists()


def test_cli_exit_codes(wired, make_collection, chroma_path, tmp_path, capsys):
    make_collection("aaa", n=20)
    assert cli.run(base(chroma_path, tmp_path)) == 2                                   # nothing selected
    assert cli.run(["migrate", "--source", "cloud", "--collection", "aaa"]) == 2          # missing credentials
    assert cli.run(base(chroma_path, tmp_path, "--collection", "missing")) == 2
    assert cli.run(base(chroma_path, tmp_path, "--collection", "aaa", "--target-collection", "ttt")) == 0
    assert cli.run(base(chroma_path, tmp_path, "--collection", "aaa", "--target-collection", "ttt")) == 2  # already completed
    wired.data["ttt"].rows.pop(map_id("doc-3"))
    assert cli.run(["validate", "--source", "local", "--chroma-path", str(chroma_path), "--workdir",
                    str(tmp_path / "wd"), "--collection", "aaa", "--target-collection", "ttt", "--poll-timeout", "0.1"]) == 1
    with pytest.raises(SystemExit) as e:
        cli.run(["migrate", "--collection", "aaa"])                                       # --source is required
    assert e.value.code == 2


def test_cli_preflight_reports_and_blocks(wired, make_collection, chroma_path, tmp_path, capsys):
    make_collection("aaa", n=20, metadata_fn=lambda i: {"document": "clash", "k": i})
    rc = cli.run(["preflight", "--source", "local", "--chroma-path", str(chroma_path), "--workdir", str(tmp_path / "wd"),
                  "--collection", "aaa"])
    rep = json.loads((tmp_path / "wd" / "aaa" / "preflight.json").read_text())
    assert rc == 0 and rep["ok"] and rep["metadata_key_renames"] == {"document": "meta_document"}
    assert any(f["code"] == "metadata_keys_renamed" for f in rep["findings"])
    rc = cli.run(["preflight", "--source", "local", "--chroma-path", str(chroma_path), "--workdir", str(tmp_path / "wd"),
                  "--collection", "aaa", "--id-strategy", "numeric"])
    assert rc == 1


def test_cli_list(wired, make_collection, chroma_path, capsys):
    make_collection("aaa", n=4, space="ip")
    assert cli.run(["list", "--source", "local", "--chroma-path", str(chroma_path)]) == 0
    assert "aaa\tcount=4\tspace=ip (configuration)" in capsys.readouterr().out
