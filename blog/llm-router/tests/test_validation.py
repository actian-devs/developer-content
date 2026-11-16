"""Frozen contract and exact order-ID validation; fixtures only."""
import json
import unittest
from llm_router.reader_config import load_config


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_config(None, environ={}).contract()
        self.answer = dict(category='refund', order_id='ORD-123', requested_action='refund',
                           urgency='normal', needs_clarification=False)

    def check(self, message, **changes):
        return self.contract.check(json.dumps(self.answer | changes), message)

    def test_known_identifier_passes_but_semantics_are_not_checked(self):
        self.assertTrue(self.check('Refund ORD-123')['runtime_check_pass'])

    def test_invented_identifier_and_partial_boundary_fail(self):
        for message in ['Refund ORD-999', 'Refund ORD-1234', 'Refund XORD-123']:
            self.assertFalse(self.check(message)['runtime_check_pass'])

    def test_clarification_consistency(self):
        self.assertFalse(self.check('Refund ORD-123', requested_action='clarify')['runtime_check_pass'])
        self.assertTrue(self.check('Refund ORD-123', requested_action='clarify',
                                   needs_clarification=True)['runtime_check_pass'])

    def test_schema_rejects_extra_fields(self):
        self.assertFalse(self.check('Refund ORD-123', unexpected='value')['schema_valid'])
