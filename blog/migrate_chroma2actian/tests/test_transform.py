import uuid

import numpy as np
import pytest
from actian_vectorai import Distance

from chroma2actian.errors import UsageError
from chroma2actian.transform import (BadRecord, IdSetDigest, IdStrategy, RecordTransformer, TransformOptions,
                                     canonical_id, chunk_points, map_id, resolve_distance)


# ---------------------------------------------------------------- IDs
def test_uuid5_is_deterministic_valid_and_distinct():
    a, b = map_id("doc-1"), map_id("doc-2")
    assert a == map_id("doc-1") and a != b
    assert str(uuid.UUID(a)) == a


def test_uuid5_namespace_changes_mapping():
    assert map_id("x") != map_id("x", namespace=uuid.UUID(int=1))


def test_preserve_keeps_canonical_uuid_and_ints_and_remaps_rest():
    u = str(uuid.uuid4())
    assert map_id(u, IdStrategy.PRESERVE) == u
    assert map_id("42", IdStrategy.PRESERVE) == 42
    assert map_id("007", IdStrategy.PRESERVE) != 7          # non-canonical int is NOT silently collapsed
    assert map_id(u.upper(), IdStrategy.PRESERVE) != u.upper()  # non-canonical uuid text -> uuid5
    assert str(uuid.UUID(map_id("hello", IdStrategy.PRESERVE)))


def test_numeric_is_strict():
    assert map_id("123", IdStrategy.NUMERIC) == 123
    for bad in ("abc", "-1", "01", "1.5", str(2**64), ""):
        with pytest.raises(BadRecord):
            map_id(bad, IdStrategy.NUMERIC)


def test_canonical_id_normalises_uuid_case_and_ints():
    u = str(uuid.uuid4())
    assert canonical_id(u.upper()) == u and canonical_id(5) == "5"


def test_digest_is_order_independent_and_collision_resistant():
    a, b, c = IdSetDigest(), IdSetDigest(), IdSetDigest()
    for i in ("1", "4"):
        a.add(i)
    for i in ("2", "3"):
        b.add(i)                    # {1,4} vs {2,3}: same count and same plain sum
    assert a != b
    for i in ("4", "1"):
        c.add(i)
    assert a == c
    d = IdSetDigest()
    d.add("1")
    assert d != a


# --------------------------------------------------------------- distance
def test_distance_mapping_and_override():
    assert resolve_distance("l2", None) == Distance.Euclid
    assert resolve_distance("cosine", "auto") == Distance.Cosine
    assert resolve_distance("ip", None) == Distance.Dot
    assert resolve_distance("l2", "dot") == Distance.Dot
    with pytest.raises(UsageError):
        resolve_distance("weird", None)


# ---------------------------------------------------------------- records
def conv(t, cid="a", emb=(1.0, 2.0), doc="d", md=None):
    return t.convert(cid, np.array(emb, dtype=np.float32), doc, md)


def test_payload_contains_metadata_document_and_original_id():
    t = RecordTransformer(TransformOptions(), dimension=2)
    p, size = conv(t, "a", md={"k": 1, "s": "x", "f": 1.5, "b": True, "arr": [1, 2]})
    assert p.payload == {"k": 1, "s": "x", "f": 1.5, "b": True, "arr": [1, 2], "chroma_id": "a", "document": "d"}
    assert size > 0
    p.to_proto()                                     # encodable by the real SDK


def test_reserved_keys_are_renamed_and_logged():
    t = RecordTransformer(TransformOptions(), dimension=2)
    p, _ = conv(t, md={"document": "meta-doc", "chroma_id": 7})
    assert p.payload["meta_document"] == "meta-doc" and p.payload["meta_chroma_id"] == 7
    assert p.payload["document"] == "d" and p.payload["chroma_id"] == "a"
    assert t.field_map == {"document": "meta_document", "chroma_id": "meta_chroma_id"}


def test_rename_collision_is_a_bad_record():
    t = RecordTransformer(TransformOptions(), dimension=2)
    with pytest.raises(BadRecord, match="collides"):
        conv(t, md={"document": 1, "meta_document": 2})


def test_none_document_omitted_and_no_documents_option():
    t = RecordTransformer(TransformOptions(), dimension=2)
    p, _ = conv(t, doc=None)
    assert "document" not in p.payload and t.counters["document_none_omitted"] == 1
    t2 = RecordTransformer(TransformOptions(include_documents=False), dimension=2)
    p2, _ = conv(t2, md={"document": "kept-as-is"})
    assert p2.payload["document"] == "kept-as-is" and "meta_document" not in p2.payload


def test_numpy_scalars_and_big_ints():
    t = RecordTransformer(TransformOptions(), dimension=2)
    p, _ = conv(t, md={"i": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True), "big": 2**70, "n": None})
    assert p.payload["i"] == 3 and type(p.payload["i"]) is int
    assert type(p.payload["f"]) is float and p.payload["b"] is True
    assert p.payload["big"] == str(2**70) and t.counters["int_out_of_int64_stringified"] == 1
    p.to_proto()                                     # would raise ValueError if the big int had slipped through


@pytest.mark.parametrize("emb,msg", [((1.0, float("nan")), "NaN"), ((1.0, float("inf")), "NaN"),
                                     ((1.0, 2.0, 3.0), "dimension"), ((), "non-empty")])
def test_bad_vectors_rejected(emb, msg):
    t = RecordTransformer(TransformOptions(), dimension=2)
    with pytest.raises(BadRecord, match=msg):
        t.convert("a", np.array(emb, dtype=np.float32), None, None)


def test_zero_vector_rejected_only_when_requested():
    with pytest.raises(BadRecord, match="zero vector"):
        conv(RecordTransformer(TransformOptions(reject_zero_vectors=True), dimension=2), emb=(0.0, 0.0))
    conv(RecordTransformer(TransformOptions(reject_zero_vectors=False), dimension=2), emb=(0.0, 0.0))


def test_unsupported_value_type_rejected():
    with pytest.raises(BadRecord, match="unsupported"):
        conv(RecordTransformer(TransformOptions(), dimension=2), md={"x": object()})


def test_chunking_respects_count_and_bytes():
    t = RecordTransformer(TransformOptions(), dimension=2)
    pts = [conv(t, f"i{i}")[0] for i in range(10)]
    assert [len(b) for b in chunk_points(pts, [1] * 10, 4, 10**9)] == [4, 4, 2]
    assert [len(b) for b in chunk_points(pts, [6] * 10, 100, 12)] == [2] * 5
    assert [len(b) for b in chunk_points(pts[:2], [100, 100], 100, 10)] == [1, 1]   # oversize point goes alone
