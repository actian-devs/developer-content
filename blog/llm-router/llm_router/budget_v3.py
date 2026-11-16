"""Exposure-inclusive accounting v3; original ledger records are retained."""
import json
from .budget import Ledger as PreviousLedger,BudgetStop,token_cost

class Ledger(PreviousLedger):
    def reserve(self,attempt_id,request_payload,max_output_tokens):
        encoded=json.dumps(request_payload,ensure_ascii=False).encode()
        if len(encoded)>32000 or not 1<=max_output_tokens<=256:raise BudgetStop('Unsupported reservation envelope')
        from .budget import PRICE_NOTE
        detail=dict(input_token_bound=len(encoded)+8192,max_output_tokens=max_output_tokens,price_assumption=PRICE_NOTE,
                    accounting_version='exposure-inclusive-v3')
        amount=token_cost(detail['input_token_bound'],max_output_tokens)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT id FROM reservations WHERE status='pending'").fetchone():
                raise BudgetStop('Active/pending request must be reconciled before continuing')
            exposure=db.execute("SELECT COALESCE(SUM(CASE WHEN status='settled' THEN actual ELSE MAX(reserved,COALESCE(actual,0)) END),0) FROM reservations").fetchone()[0]
            cap=db.execute('SELECT amount FROM budget_limit WHERE id=1').fetchone()[0]
            if exposure+amount>cap:raise BudgetStop('Confirmed charges + full uncertain exposure + next reservation exceed cap')
            db.execute('INSERT INTO reservations VALUES (?,?,?,?,?)',(attempt_id,'pending',amount,None,json.dumps(detail)))
        return dict(reserved_microusd=amount,**detail)

    def summary(self):
        result=super().summary()
        result.update(accounting_version='exposure-inclusive-v3',blocked=any(r['status']=='pending' for r in result['entries']),
            unresolved_exposure_present=any(r['status']!='settled' for r in result['entries']))
        result['available_after_exposure_microusd']=result['ceiling_microusd']-result['settled_microusd']-result['unresolved_reserved_microusd']
        return result
