import time
import uuid
from datetime import datetime, timezone
from .contract import assert_runtime_only, message_key, fingerprint
from .providers import ProviderFailure
from .routing import decide


class ExecutionStop(RuntimeError):
    pass


class Engine:
    def __init__(self, contract, local, cloud, journal, store, config_id):
        self.contract, self.local, self.cloud = contract, local, cloud
        self.journal, self.store, self.config_id = journal, store, config_id

    def attempt(self, message, message_id, request_id, model_kind, stage, namespace, vector, decision, eligible_seed=False):
        record = dict(attempt_id=str(uuid.uuid4()), request_id=request_id, message_id=message_id,
                      message=message, message_key=message_key(message), model_kind=model_kind,
                      stage=stage, namespace=namespace, eligible_seed=eligible_seed and stage=='seed',
                      semantic_outcome='unverified',
                      validator_version='runtime-validator-v2.0.0',
                      config_id=self.config_id, contract_version=self.contract.version,
                      requested_model=(self.contract.settings['local_model'] if model_kind=='local' else getattr(self.cloud,'model',self.contract.settings['openai_model'])),
                      prompt_sha256=self.contract.prompt_hash, local_manifest_digest=self.contract.settings['local_manifest_digest'],
                      embedding_manifest_digest=self.contract.settings['embedding_manifest_digest'],
                      inference_settings=self.contract.settings, decision=decision,
                      execution_controls=dict(paid_enabled=bool(getattr(self.cloud,'enable_paid',False)),persistent_budget_ceiling_usd=(self.cloud.ledger.summary()['ceiling_microusd']/1e6 if hasattr(self.cloud,'ledger') else .75),
                                              runtime_check_is_not_semantic_correctness=True,sdk_cloud_retries=0),
                      started_at=datetime.now(timezone.utc).isoformat(),status='started')
        self.journal.put(record)  # Persist identity before any provider request.
        start = time.perf_counter(); fatal = None
        try:
            result = (self.local if model_kind=='local' else self.cloud).generate(message, record['attempt_id'])
            record.update(result)
            record.update(self.contract.check(result.get('text',''),message))
            if not result.get('completion_ok',False):
                record['runtime_check_pass'] = False
                record['runtime_errors'].append('Provider did not complete a bounded response')
            record['status'] = 'completed'
        except Exception as exc:
            detail = getattr(exc,'detail',{})
            record.update(detail)
            record.update(status='error', schema_valid=False, runtime_check_pass=False, parsed=None,
                          runtime_errors=[str(exc)], generation_attempted=detail.get('generation_attempted',False))
            if getattr(exc,'fatal',False) or not isinstance(exc,ProviderFailure):
                fatal = str(exc)
        record.update(latency_seconds=time.perf_counter()-start, finished_at=datetime.now(timezone.utc).isoformat())
        self.journal.put(record)
        # Even a failed generation attempt is retained, under its own UUID.
        self.store.write(record, vector)
        if fatal:
            raise ExecutionStop(fatal)
        return record

    def route(self, message, message_id, snapshot, policy, namespace, vector=None, stage='evaluation'):
        vector = vector if vector is not None else self.local.embed(message)
        start = time.perf_counter()
        semantic='feedback' in snapshot
        hits = self.store.search(vector, list(snapshot['feedback'] if semantic else snapshot['evidence']))
        if semantic:
            from .semantic_history import decide_semantic
            decision=decide_semantic(hits,snapshot,message_key(message),policy.similarity_min)
        elif 'runtime_revalidations' in snapshot:
            from .semantic_history import decide_runtime_comparison
            decision=decide_runtime_comparison(hits,snapshot['runtime_revalidations'],snapshot['config_id'],message_key(message),policy.similarity_min)
        else:decision = decide(hits, snapshot, message_key(message), policy)
        decision.update(snapshot_id=snapshot['snapshot_id'], similarity_min=policy.similarity_min,
                        search_latency_seconds=time.perf_counter()-start,
                        retrieved_hits=[dict(id=str(h['id']),score=h['score'],message_key=h['payload']['message_key']) for h in hits])
        request_id = str(uuid.uuid4())
        first = self.attempt(message,message_id,request_id,decision['selected_model'],stage,namespace,vector,decision)
        attempts = [first]
        if decision['selected_model']=='local' and not first['runtime_check_pass']:
            escalation = dict(decision, escalation_reason='local_runtime_check_failed', local_attempt_id=first['attempt_id'])
            attempts.append(self.attempt(message,message_id,request_id,'cloud',stage,namespace,vector,escalation))
        return dict(request_id=request_id,message_id=message_id,decision=decision,initial_attempt_id=attempts[0]['attempt_id'],
                    final_attempt_id=attempts[-1]['attempt_id'],attempt_ids=[r['attempt_id'] for r in attempts],
                    initial=attempts[0],final=attempts[-1],escalated=len(attempts)>1)


def freeze_snapshot(records, message_ids, config_id):
    selected = [r for r in records if r.get('stage')=='seed' and r.get('model_kind')=='local'
                and r.get('eligible_seed') is True and r.get('generation_attempted') is True
                and r.get('config_id')==config_id and r['message_id'] in message_ids and r.get('status')!='started']
    if {r['message_id'] for r in selected} != set(message_ids):
        raise ValueError('Checkpoint missing dispatched local seed evidence for one or more manifest messages')
    for r in selected: assert_runtime_only(r)
    snap = dict(config_id=config_id,message_ids=sorted(message_ids),evidence={r['attempt_id']:fingerprint(r) for r in selected})
    return dict(snapshot_id=fingerprint(snap), **snap)
