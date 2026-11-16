"""Seed-only draft-feedback ingestion and immutable semantic routing projections.
No label files or expected answers are opened by this runtime module.
"""
import math,hashlib
from .contract import fingerprint,message_key,assert_runtime_only

VERSION='semantic-seed-history-v1.1.0'
MATCH='draft_reference_match'
MISMATCH='draft_reference_mismatch'

def import_feedback(record,attempt,allowlist,configuration):
    assert_runtime_only(record)
    mid=record['message_id']
    if mid not in allowlist or record.get('split')!='seed' or attempt.get('stage')!='seed' or attempt.get('model_kind')!='local':raise ValueError('Only allowlisted local seed attempts may receive feedback')
    expected=allowlist[mid]
    if attempt['message_id']!=mid or attempt['config_id']!=configuration:raise ValueError('Wrong message/configuration')
    for key,value in dict(attempt_id=attempt['attempt_id'],input_hash=message_key(attempt['message']),input_sha256=hashlib.sha256(attempt['message'].encode()).hexdigest(),attempt_hash=fingerprint(attempt),config_id=configuration).items():
        if record.get(key)!=value:raise ValueError('Feedback binding mismatch: '+key)
    if expected['input_hash']!=record['input_hash'] or expected['attempt_id']!=record['attempt_id']:raise ValueError('Seed allowlist mismatch')
    if expected.get('input_sha256')!=record['input_sha256'] or expected.get('attempt_hash')!=record['attempt_hash']:raise ValueError('Exact input/attempt allowlist hash mismatch')
    if record['outcome'] not in [MATCH,MISMATCH]:raise ValueError('Only explicit draft outcomes accepted at seed import')
    return dict(record)

def decide_semantic(hits,snapshot,query_key,boundary,excluded_family=None):
    assert_runtime_only(hits)
    if fingerprint({k:v for k,v in snapshot.items() if k!='snapshot_id'})!=snapshot['snapshot_id']:raise ValueError('Snapshot drift')
    if boundary is None:return dict(selected_model='cloud',reason='no_seed_calibration_candidate_qualified',neighbors=[])
    required={aid for aid,f in snapshot['feedback'].items() if f['input_hash']!=query_key and f['scenario_family_id']!=excluded_family and f['outcome'] in [MATCH,MISMATCH]}
    if not required<={str(h['id']) for h in hits}:return dict(selected_model='cloud',reason='incomplete_historical_retrieval',neighbors=[])
    groups={}
    for hit in hits:
        aid=str(hit['id'])
        if aid not in snapshot['feedback']:continue # Live/unverified is not support.
        evidence=snapshot['feedback'][aid]
        if fingerprint(hit['payload'])!=evidence['attempt_hash']:raise ValueError('Attempt drift')
        if evidence['input_hash']==query_key or evidence['scenario_family_id']==excluded_family:continue
        if evidence['outcome'] not in [MATCH,MISMATCH]:continue
        score=float(hit['score'])
        if not math.isfinite(score):continue
        g=groups.setdefault(evidence['input_hash'],dict(score=score,match=True,attempt_ids=[]))
        g['score']=max(g['score'],score);g['match'] &= evidence['outcome']==MATCH;g['attempt_ids'].append(aid)
    neighbors=sorted((g for g in groups.values() if g['score']>=boundary),key=lambda g:-g['score'])[:10]
    n=len(neighbors);matching=sum(g['match'] for g in neighbors)
    local=n>=5 and matching*5>=n*4
    return dict(selected_model='local' if local else 'cloud',reason='exploratory_draft_history_gate' if local else 'insufficient_draft_match_support',neighbors=neighbors,distinct_support=n,historical_match_rate=matching/n if n else None)

def freeze_feedback(records,message_ids):
    selected={r['attempt_id']:r for r in records if r['message_id'] in message_ids}
    if {r['message_id'] for r in selected.values()}!=set(message_ids):raise ValueError('Incomplete seed checkpoint')
    value=dict(version=VERSION,message_ids=sorted(message_ids),feedback=selected)
    return dict(value,snapshot_id=fingerprint(value))

def decide_runtime_comparison(hits,revalidations,configuration,query_key,boundary):
    """Old policy with new operational validator results; no semantic outcomes."""
    from .routing import decide,RoutingPolicy
    projected=[]
    for h in hits:
        aid=str(h['id']);validation=revalidations[aid]
        if fingerprint(h['payload'])!=validation['attempt_hash']:raise ValueError('Original attempt changed')
        p=dict(h['payload'],runtime_check_pass=validation['new_validation']['runtime_check_pass'])
        projected.append(dict(h,payload=p))
    snapshot=dict(config_id=configuration,evidence={str(h['id']):fingerprint(h['payload']) for h in projected})
    return decide(projected,snapshot,query_key,RoutingPolicy(boundary))
