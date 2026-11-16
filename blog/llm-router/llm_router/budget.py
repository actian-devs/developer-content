"""One persistent experiment-wide USD ledger. Reservations survive crashes."""
import json
import sqlite3
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from contextlib import contextmanager

MICROS = 1_000_000
CEILING = 750_000
INPUT_USD_PER_M = Decimal('2.50')
OUTPUT_USD_PER_M = Decimal('15.00')
PRICE_NOTE = 'GPT-5.4 standard assumed USD 2.50/M input, 15/M output; no cache discount, tools, regional uplift or taxes. Verify before paid use.'


class BudgetStop(RuntimeError):
    pass


def token_cost(input_tokens, output_tokens):
    if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
        raise BudgetStop('Missing/invalid usage; do not infer zero cost')
    return int((Decimal(input_tokens)*INPUT_USD_PER_M + Decimal(output_tokens)*OUTPUT_USD_PER_M).to_integral_value(rounding=ROUND_CEILING))


class Ledger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, status TEXT NOT NULL, reserved INTEGER NOT NULL, actual INTEGER, detail TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS budget_limit (id INTEGER PRIMARY KEY CHECK(id=1), amount INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS cap_history (old_amount INTEGER,new_amount INTEGER,reason TEXT)')
            if not db.execute('SELECT amount FROM budget_limit WHERE id=1').fetchone():
                prior=3*MICROS if db.execute('SELECT COUNT(*) FROM reservations').fetchone()[0] else CEILING
                db.execute('INSERT INTO budget_limit VALUES (1,?)',(CEILING,))
                db.execute('INSERT INTO cap_history VALUES (?,?,?)',(prior,CEILING,'User-authorized cumulative cap reduction for baseline-and-replay experiment'))

    def lower_cap(self, amount):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            current=db.execute('SELECT amount FROM budget_limit WHERE id=1').fetchone()[0]
            exposure=db.execute("SELECT COALESCE(SUM(CASE WHEN status='settled' THEN actual ELSE MAX(reserved,COALESCE(actual,0)) END),0) FROM reservations").fetchone()[0]
            if amount>current or amount<exposure:raise BudgetStop('Cannot raise cap or lower below already-accounted exposure')
            if amount<current:
                db.execute('UPDATE budget_limit SET amount=? WHERE id=1',(amount,));db.execute('INSERT INTO cap_history VALUES (?,?,?)',(current,amount,'Explicit cap reduction'))

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, attempt_id, request_payload, max_output_tokens):
        # UTF-8 bytes upper-bound byte-BPE text tokens; generous framing/schema margin.
        # No remote tools, files, images or history; payload size and model are restricted.
        encoded = json.dumps(request_payload, ensure_ascii=False).encode()
        if len(encoded) > 32_000 or not 1 <= max_output_tokens <= 256:
            raise BudgetStop('Request exceeds the supported conservative reservation envelope')
        input_bound = len(encoded) + 8192
        amount = token_cost(input_bound, max_output_tokens)
        detail = dict(input_token_bound=input_bound, max_output_tokens=max_output_tokens, price_assumption=PRICE_NOTE)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            pending = db.execute("SELECT id FROM reservations WHERE status IN ('pending','uncertain')").fetchall()
            if pending:
                raise BudgetStop('Unreconciled billed exposure exists; manual billing reconciliation required: '+str(pending))
            spent = db.execute("SELECT COALESCE(SUM(actual),0) FROM reservations WHERE status='settled'").fetchone()[0]
            cap=db.execute('SELECT amount FROM budget_limit WHERE id=1').fetchone()[0]
            if spent + amount > cap:
                raise BudgetStop('Persistent experiment ceiling: reservation refused')
            db.execute('INSERT INTO reservations VALUES (?,?,?,?,?)', (attempt_id, 'pending', amount, None, json.dumps(detail)))
        return dict(reserved_microusd=amount, **detail)

    def uncertain(self, attempt_id, reason):
        with self.db() as db:
            db.execute("UPDATE reservations SET status='uncertain', detail=json_set(detail,'$.uncertainty',?) WHERE id=? AND status='pending'", (str(reason), attempt_id))

    def settle(self, attempt_id, input_tokens, output_tokens):
        try:
            actual = token_cost(input_tokens, output_tokens)
        except BudgetStop:
            self.uncertain(attempt_id, 'Usage not safely reconcilable')
            raise
        over = False
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status,reserved FROM reservations WHERE id=?', (attempt_id,)).fetchone()
            if not row or row[0] != 'pending':
                raise BudgetStop('Reservation missing or already reconciled')
            over = actual > row[1]
            db.execute('UPDATE reservations SET status=?,actual=? WHERE id=?', ('uncertain' if over else 'settled', actual, attempt_id))
        if over:
            raise BudgetStop('Reported cost exceeded conservative reservation; stop for billing reconciliation')
        return actual

    def summary(self):
        with self.db() as db:
            rows = db.execute('SELECT id,status,reserved,actual,detail FROM reservations ORDER BY rowid').fetchall()
            cap=db.execute('SELECT amount FROM budget_limit WHERE id=1').fetchone()[0]
            history=db.execute('SELECT old_amount,new_amount,reason FROM cap_history').fetchall()
        return dict(ceiling_microusd=cap,cap_history=history, settled_microusd=sum(r[3] or 0 for r in rows if r[1]=='settled'),
                    unresolved_reserved_microusd=sum(max(r[2],r[3] or 0) for r in rows if r[1]!='settled'),
                    blocked=any(r[1]!='settled' for r in rows), entries=[dict(attempt_id=r[0], status=r[1], reserved_microusd=r[2], actual_microusd=r[3], detail=json.loads(r[4])) for r in rows])
