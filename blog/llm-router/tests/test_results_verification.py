"""Offline export/aggregation tests. No inference, database or network calls."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1] / 'results'
SPEC = importlib.util.spec_from_file_location('results_verifier', BASE / 'verify.py')
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class PublishedResultsTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((BASE / 'recorded_results.json').read_text())

    def test_published_figures_without_network(self):
        with patch('socket.socket', side_effect=AssertionError('Network forbidden')):
            summary = VERIFY.verify_reported_figures(self.data)
        self.assertEqual(summary['api_only_calculation']['break_even_whole_requests'], 1987)

    def test_duplicate_ids_are_rejected(self):
        self.data['evaluation'][-1]['id'] = 'E-001'
        with self.assertRaises(ValueError):
            VERIFY.compute(self.data)

    def test_cost_or_correctness_tampering_is_detected(self):
        changed = copy.deepcopy(self.data)
        changed['cloud_seeding'][0]['api_cost_microusd'] += 1
        with self.assertRaises(ValueError):
            VERIFY.verify_reported_figures(changed)
        changed = copy.deepcopy(self.data)
        changed['evaluation'][0]['local']['draft_reference_match'] = not changed['evaluation'][0]['local']['draft_reference_match']
        with self.assertRaises(ValueError):
            VERIFY.verify_reported_figures(changed)

    def test_runtime_rejection_includes_both_attempts_on_escalation(self):
        row = next(r for r in self.data['evaluation'] if r['id'] == 'E-041')
        row['local']['runtime_check_pass'] = False
        result = VERIFY.compute(self.data)['checkpoints']['40']
        self.assertEqual(result['escalation_ids'], ['E-041'])
        cloud = row['cloud']['api_cost_microusd']
        self.assertEqual(result['counterfactual_replay_api_cost_microusd'], 281686 + cloud)
        with self.assertRaises(ValueError):
            VERIFY.verify_reported_figures(self.data)

    def test_missing_cloud_is_not_a_free_response(self):
        self.assertIsNone(self.data['evaluation'][0]['cloud'])
        summary = VERIFY.compute(self.data)
        for result in summary['checkpoints'].values():
            self.assertEqual(result['completed'], 99)
            self.assertEqual(result['missing_ids'], ['E-001'])
