import tempfile,unittest
from pathlib import Path
from llm_router.budget_v3 import Ledger,BudgetStop

class ExposureTests(unittest.TestCase):
    def test_uncertain_exposure_preserved_and_new_request_fits(self):
        with tempfile.TemporaryDirectory() as d:
            l=Ledger(Path(d)/'ledger');r=l.reserve('uncertain',{},256);l.uncertain('uncertain','fixture')
            l.reserve('next',{},256);l.settle('next',1000,50)
            s=l.summary();self.assertEqual(s['unresolved_reserved_microusd'],r['reserved_microusd'])
            self.assertEqual(s['entries'][0]['status'],'uncertain');self.assertFalse(s['blocked'])
            l.lower_cap(s['settled_microusd']+s['unresolved_reserved_microusd'])
            with self.assertRaises(BudgetStop):l.reserve('over',{},256)
    def test_active_pending_request_and_duplicate_id_are_not_resent(self):
        with tempfile.TemporaryDirectory() as d:
            l=Ledger(Path(d)/'ledger');l.reserve('first',{},256)
            with self.assertRaises(BudgetStop):l.reserve('other',{},256)
            l.uncertain('first','fixture')
            with self.assertRaises(Exception):l.reserve('first',{},256)
