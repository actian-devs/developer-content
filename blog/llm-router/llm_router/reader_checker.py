"""Separate OFFLINE bounded-example checker; never imported by route/provider code."""
from .reader_tutorial import DATA,read,save,inputs
from .storage import Journal
from .contract import fingerprint,message_key
from .outcome_loop import make_feedback

def check_run(work,c,split):
    rows=inputs(split);by={r['id']:r for r in rows};labels={r['id']:r['reference'] for r in read(DATA/'offline_labels'/(split+'.json'))}
    journal=Journal(work/'attempts.sqlite3');records=[a for a in journal.all() if a['stage']==split]
    allow={r['id']:message_key(r['message']) for r in rows};feedback=[];audit=[]
    for a in records:
        if a['message_id'] not in by or a['message']!=by[a['message_id']]['message']:raise ValueError('Input binding changed')
        if a['status']!='completed':raise ValueError('Interrupted attempt needs reconciliation before checking')
        if a['contract_version']!=c.version or a['prompt_sha256']!=c.prompt_hash:raise ValueError('Contract changed')
        check=c.check(a['text'],a['message']);expected=labels[a['message_id']]
        entry=dict(attempt_id=a['attempt_id'],expected=expected,actual=check['parsed'],passed=check['parsed']==expected,
            provenance='Synthetic draft exact agreement; not independent human review')
        audit.append(entry)
        feedback.append(make_feedback(a,entry['passed'],dict(kind='synthetic_draft_reference_exact_match',evidence_sha256=fingerprint(entry),
            human_reviewed=False,scope='reader_bounded_example_'+split),allow))
    path=work/'checked-feedback.json';old=read(path) if path.exists() else [];merged={f['feedback_id']:f for f in old+feedback}
    save(work/'offline-checks'/(split+'.json'),audit);save(path,list(merged.values()))
    return dict(checked=len(records),matches=sum(a['passed'] for a in audit),check='Synthetic draft exact agreement, not verified correctness')
