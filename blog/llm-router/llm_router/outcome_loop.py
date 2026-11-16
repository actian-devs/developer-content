"""Versioned outcome-feedback loop. No references or model calls in routing.
Native success retrieval is separate from failure-inclusive reliability evidence.
"""
import uuid,math
from actian_vectorai import FilterBuilder,Field,has_id,SearchParams,PointStruct
from .contract import fingerprint,message_key,assert_runtime_only
from .storage import COLLECTION,vector_check
from .native_history_store import NativeHistoryStore

VERSION='outcome-feedback-loop-v1.0.0'

def make_feedback(attempt,passed,check_binding,allowlist):
    """Explicit checker boundary; caller supplies a checked result, never a model guess.
    Draft-reference feedback is permitted only for the disclosed seed/demo example.
    Production callers must supply reviewed feedback and retain its audit source.
    """
    mid=attempt['message_id']
    if mid not in allowlist or allowlist[mid]!=message_key(attempt['message']):raise ValueError('Feedback ID/input is not allowlisted')
    stage=attempt['stage']
    if stage not in ['seed','demo','live'] or mid.startswith('E-'):raise ValueError('Evaluation feedback forbidden')
    if stage=='seed' and not mid.startswith('S-') or stage=='demo' and not mid.startswith('D-'):raise ValueError('Wrong feedback scope')
    if type(passed) is not bool:raise ValueError('Feedback requires explicit boolean result')
    if check_binding.get('kind') not in ['synthetic_draft_reference_exact_match','human_review']:raise ValueError('Runtime validation/model output is not semantic feedback')
    if not check_binding.get('evidence_sha256'):raise ValueError('Checker audit binding required')
    if stage=='live' and check_binding['kind']!='human_review':raise ValueError('Unreviewed live outcomes must stay unverified')
    value=dict(record_kind='outcome_feedback',version=VERSION,namespace='article-feedback-loop-v1',message_id=mid,
        input_hash=message_key(attempt['message']),source_attempt_id=attempt['attempt_id'],source_attempt_hash=fingerprint(attempt),
        model_kind=attempt['model_kind'],passed_check=passed,check_binding=check_binding,
        semantic_status=('draft_reference_match' if passed else 'draft_reference_mismatch') if check_binding['kind'].startswith('synthetic') else ('reviewed_match' if passed else 'reviewed_mismatch'),
        runtime_check_pass=attempt['runtime_check_pass'],source_api_cost_microusd=attempt.get('actual_cost_microusd',0),
        local_manifest_digest=attempt['local_manifest_digest'],prompt_sha256=attempt['prompt_sha256'])
    value['feedback_id']=str(uuid.uuid5(uuid.NAMESPACE_URL,VERSION+':'+fingerprint(value)))
    assert_runtime_only(value);return value

class OutcomeStore(NativeHistoryStore):
    def upsert_feedback(self,feedback,vector):
        assert_runtime_only(feedback);vector_check(vector);fid=feedback['feedback_id']
        if feedback.get('record_kind')!='outcome_feedback' or feedback.get('version')!=VERSION or type(feedback.get('passed_check')) is not bool:
            raise ValueError('Explicit versioned semantic feedback required')
        if feedback.get('check_binding',{}).get('kind') not in ['synthetic_draft_reference_exact_match','human_review']:
            raise ValueError('Runtime checks are not semantic feedback')
        expected_id=str(uuid.uuid5(uuid.NAMESPACE_URL,VERSION+':'+fingerprint({k:v for k,v in feedback.items() if k!='feedback_id'})))
        if fid!=expected_id:raise ValueError('Feedback content binding changed')
        source=self.read([feedback['source_attempt_id']])
        if len(source)!=1 or fingerprint(source[0]['payload'])!=feedback['source_attempt_hash']:
            raise ValueError('Source attempt missing or changed')
        stage=source[0]['payload']['stage']
        if stage not in ['seed','demo','live'] or source[0]['payload']['message_id'].startswith('E-'):raise ValueError('Evaluation feedback forbidden')
        if stage=='live' and feedback['check_binding']['kind']!='human_review':raise ValueError('Live semantic feedback must be reviewed')
        old=self.read([fid])
        if old:
            if fingerprint(old[0]['payload'])!=fingerprint(feedback):raise ValueError('Immutable feedback conflict')
            return
        self.client.points.upsert(getattr(self,'collection',COLLECTION),[PointStruct(id=fid,vector=vector,payload=feedback)],timeout=15)
        got=self.read([fid])
        if len(got)!=1 or fingerprint(got[0]['payload'])!=fingerprint(feedback):raise RuntimeError('Feedback readback failed')

    def search_successes(self,vector,feedback_records):
        """Brief's successful-outcome query. Never a success-rate denominator."""
        vector_check(vector);ids=[r['feedback_id'] for r in feedback_records]
        if not ids:return []
        expected={r['feedback_id'] for r in feedback_records if r['passed_check']}
        collection=getattr(self,'collection',COLLECTION)
        count=int(self.client.collections.get_info(collection).model_dump(mode='json')['points_count'])
        if count>10000:raise ValueError('Native workaround not verified above 10000 points')
        filt=FilterBuilder().must(has_id(ids)).must(Field('passed_check').eq(True)).build()
        rows=self.client.points.search(collection,vector,filter=filt,limit=max(1,count),
            params=SearchParams(exact=True,hnsw_ef=max(4096,count)),with_payload=True,timeout=30)
        hits=[r.model_dump(mode='json') for r in rows]
        actual=[str(h['id']) for h in hits]
        if set(actual)!=expected or len(actual)!=len(expected) or any(h['payload']['passed_check'] is not True for h in hits):
            raise RuntimeError('Native passed-check filter did not return the expected successful outcomes')
        return hits

def choose(all_hits,successful_hits,query_hash,boundary=.5):
    """Reliability uses all checked outcomes. Same five / 80% / top-ten settings."""
    if boundary!=.5:raise ValueError('This version freezes the existing 0.5 boundary')
    all_by={str(h['id']):h for h in all_hits}
    if any(str(h['id']) not in all_by for h in successful_hits):raise ValueError('Successful query outside evidence')
    tiers={}
    for kind in ['local','cloud']:
        groups={}
        for h in all_hits:
            p=h['payload'];score=float(h['score'])
            if p['model_kind']!=kind or p['input_hash']==query_hash or not math.isfinite(score) or score<boundary:continue
            g=groups.setdefault(p['input_hash'],dict(score=score,passed=True,feedback_ids=[]))
            g['score']=max(score,g['score']);g['passed'] &= p['passed_check'];g['feedback_ids'].append(str(h['id']))
        neighbors=sorted(groups.values(),key=lambda g:-g['score'])[:10]
        n=len(neighbors);passed=sum(g['passed'] for g in neighbors)
        success_ids={str(h['id']) for h in successful_hits if h['payload']['model_kind']==kind}
        qualifying=n>=5 and passed*5>=n*4 and any(set(g['feedback_ids'])&success_ids for g in neighbors)
        tiers[kind]=dict(distinct_support=n,passed=passed,failed=n-passed,match_rate=passed/n if n else None,qualifies=qualifying,neighbors=neighbors)
    local=tiers['local']['qualifies']
    return dict(selected_model='local' if local else 'cloud',reason='checked_local_history_gate' if local else 'cloud_fallback_insufficient_checked_local_support',
        policy_version=VERSION,similarity_min=.5,minimum_distinct_messages=5,match_threshold=.8,tiers=tiers)

def route(store,vector,feedback_records,query):
    # Both queries use native points.search. No labels, Python cosine or success-only rate.
    configurations={(r['local_manifest_digest'],r['prompt_sha256']) for r in feedback_records}
    if len(configurations)>1:raise ValueError('Do not mix model/prompt configurations in outcome history')
    all_hits=store.search(vector,[r['feedback_id'] for r in feedback_records])
    successful=store.search_successes(vector,feedback_records)
    expected={r['feedback_id']:fingerprint(r) for r in feedback_records}
    for h in all_hits:
        if fingerprint(h['payload'])!=expected[str(h['id'])]:raise ValueError('Feedback changed')
    decision=choose(all_hits,successful,message_key(query))
    decision.update(all_outcomes=[dict(id=str(h['id']),score=h['score'],model_kind=h['payload']['model_kind'],passed_check=h['payload']['passed_check']) for h in all_hits],
        successful_outcomes=[dict(id=str(h['id']),score=h['score']) for h in successful])
    return decision

def execute(engine,store,vector,feedback_records,message,message_id,namespace,check_callback=None,stage='live'):
    """Runnable route/attempt/check/upsert loop; provider permissions stay in Engine.
    An optional explicit checker returns a bound feedback record, never expected JSON.
    With no semantic feedback the raw attempt stays unverified and adds no support.
    """
    decision=route(store,vector,feedback_records,message);request_id=str(uuid.uuid4());attempts=[];written=[]
    def attempt(kind):
        a=engine.attempt(message,message_id,request_id,kind,stage,namespace,vector,decision)
        attempts.append(a);f=check_callback(a) if check_callback else None
        if f is not None:
            if f['source_attempt_id']!=a['attempt_id'] or f['source_attempt_hash']!=fingerprint(a):raise ValueError('Feedback not bound to exact attempt')
            store.upsert_feedback(f,vector);written.append(f)
        return a,f
    first,feedback=attempt(decision['selected_model'])
    if decision['selected_model']=='local' and (not first['runtime_check_pass'] or (feedback is not None and not feedback['passed_check'])):
        attempt('cloud') # One escalation; cloud output is not correctness feedback.
    return dict(decision=decision,attempts=attempts,attempt_ids=[a['attempt_id'] for a in attempts],request_id=request_id,
        escalated=len(attempts)>1,feedback_records=written,final=attempts[-1])
