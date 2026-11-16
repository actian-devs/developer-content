"""Deterministic SDK request fixtures; no database or model calls."""
import unittest
from unittest.mock import Mock
from actian_vectorai import FilterBuilder, has_id, SearchParams
from llm_router.native_history_store import NativeHistoryStore


class NativeRetrievalTests(unittest.TestCase):
    def store(self, points):
        store = object.__new__(NativeHistoryStore)
        store.collection = 'fixture'
        store.client = Mock()
        store.search_audit = []
        store.client.collections.get_info.return_value.model_dump.return_value = {'points_count': 500}
        store.client.points.search.return_value = [Mock(model_dump=Mock(return_value=p)) for p in points]
        return store

    def test_native_search_preserves_failure_and_sets_breadth(self):
        store = self.store([{'id': 'failed', 'payload': {'runtime_check_pass': False}, 'score': .8}])
        hits = store.search([1.] + [0.] * 767, ['failed'])
        kwargs = store.client.points.search.call_args.kwargs
        self.assertEqual(kwargs['limit'], 500)
        self.assertEqual(kwargs['params'].hnsw_ef, 4096)
        self.assertTrue(kwargs['params'].exact)
        self.assertFalse(hits[0]['payload']['runtime_check_pass'])
        store.client.points.get.assert_not_called()

    def test_incomplete_extra_or_duplicate_hits_fail_closed(self):
        for points in [[], [{'id': 'other'}], [{'id': 'a'}, {'id': 'a'}]]:
            with self.assertRaises(RuntimeError):
                self.store(points).search([1.] + [0.] * 767, ['a'])

    def test_sdk_filter_and_parameters_serialize(self):
        identifier = '00000000-0000-0000-0000-000000000001'
        proto = FilterBuilder().must(has_id([identifier])).build().to_proto()
        self.assertEqual(proto.must[0].has_id.has_id[0].uuid, identifier)
        params = SearchParams(exact=True, hnsw_ef=4096).to_proto()
        self.assertTrue(params.exact)
        self.assertEqual(params.hnsw_ef, 4096)
