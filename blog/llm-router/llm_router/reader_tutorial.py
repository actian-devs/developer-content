"""Fresh reader CLI. No historical run IDs, snapshots, journals or labels in routing."""
import argparse,json,re,uuid,os
from pathlib import Path
from .contract import ROOT,fingerprint,assert_runtime_only
from .reader_config import load_config
from .storage import Journal,vector_check
from .reader_store import ReaderStore
from .collection_setup import ensure_collection
from .providers import LocalProvider,CloudProvider
from .budget_v3 import Ledger
from .engine import Engine
from .outcome_loop import route,execute

DATA=ROOT/'tutorial_data'
VERSION='reader-tutorial-v1.0.0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)
def inputs(split):
    rows=read(DATA/'inputs'/(split+'.json'))
    if any(set(r)!={'id','message'} for r in rows):raise ValueError('Runtime inputs must have only ID/message')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate message IDs')
    return rows
def bind(work,c,config):
    value=dict(version=VERSION,contract_hash=c.hash,profile=config.profile,endpoint=config.endpoint,collection=config.collection,
        input_hashes={split:fingerprint(inputs(split)) for split in ['seed','demo']})
    path=work/'configuration.json'
    if path.exists() and read(path)!=value:raise ValueError('Run configuration changed; preserve this run and use a new run ID')
    if not path.exists():save(path,value)
def vector_for(work,row,c,local):
    path=work/'vectors'/(row['id']+'.json');identity=dict(input_hash=fingerprint(row['message']),embedding_digest=c.settings['embedding_manifest_digest'])
    if path.exists():
        v=read(path)
        if v['identity']!=identity or fingerprint(v['vector'])!=v['hash']:raise ValueError('Vector cache binding mismatch')
        vector_check(v['vector']);return v['vector']
    vector=local.embed(row['message']);save(path,dict(identity=identity,vector=vector,hash=fingerprint(vector)));return vector
def cached_attempt(journal,mid,kind,stage):
    records=[r for r in journal.all() if r['message_id']==mid and r['model_kind']==kind and r['stage']==stage]
    if len(records)>1 or records and records[0]['status']!='completed':raise ValueError('Prior interrupted/error attempt requires reconciliation; no automatic resend')
    return records[0] if records else None
def restore(store,a,v):
    got=store.read([a['attempt_id']])
    if not got:store.write(a,v)
    elif fingerprint(got[0]['payload'])!=fingerprint(a):raise ValueError('Stored attempt differs from journal')
def seed(work,c,engine,models,rows=None):
    rows=inputs('seed') if rows is None else rows;done=[]
    for row in rows:
        v=vector_for(work,row,c,engine.local)
        for kind in models:
            a=cached_attempt(engine.journal,row['id'],kind,'seed')
            if a is None:
                request_id=str(uuid.uuid5(uuid.NAMESPACE_URL,str(work)+':seed:'+row['id']))
                a=engine.attempt(row['message'],row['id'],request_id,kind,'seed',work.name+':seed',v,
                    dict(selected_model=kind,reason='fresh_paired_seeding_not_routing_feedback'))
            restore(engine.store,a,v);done.append(a['attempt_id'])
            save(work/'seed-progress.json',dict(attempt_ids=done))
    return dict(completed_attempts=len(done),models=list(models))
def feedback(work):
    path=work/'active-feedback.json';return read(path) if path.exists() else []
def demo(work,c,engine,rows=None):
    results=[];rows=inputs('demo') if rows is None else rows
    for row in rows:
        path=work/'demo'/(row['id']+'.json')
        if path.exists():results.append(read(path));continue
        if any(a['message_id']==row['id'] and a['stage']=='demo' for a in engine.journal.all()):raise ValueError('Interrupted demo requires reconciliation; do not infer twice')
        v=vector_for(work,row,c,engine.local)
        execution=execute(engine,engine.store,v,feedback(work),row['message'],row['id'],work.name+':demo',stage='demo')
        decision=execution['decision'];attempts=execution['attempts']
        result=dict(message_id=row['id'],decision=decision,attempt_ids=[a['attempt_id'] for a in attempts],
            initial=attempts[0],final=attempts[-1],escalated=len(attempts)>1,semantic_feedback='unverified',
            running_api_cost_usd=sum(a.get('actual_cost_microusd',0) for a in engine.journal.all())/1e6)
        save(path,result);results.append(result)
    return dict(completed=len(results),results=results)
def apply_feedback(work,store,journal):
    # This file contains only bound booleans/provenance, never reference answers.
    records=read(work/'checked-feedback.json');active={f['feedback_id']:f for f in feedback(work)}
    attempts={a['attempt_id']:a for a in journal.all()}
    for f in records:
        assert_runtime_only(f);a=attempts[f['source_attempt_id']]
        if fingerprint(a)!=f['source_attempt_hash']:raise ValueError('Feedback source changed')
        v=read(work/'vectors'/(a['message_id']+'.json'))['vector'];store.upsert_feedback(f,v);active[f['feedback_id']]=f
    save(work/'active-feedback.json',list(active.values()));return dict(checked_outcomes=len(active))
def parser():
    p=argparse.ArgumentParser(description='Fresh native tutorial; paid inference disabled unless explicitly enabled')
    p.add_argument('--env-file',type=Path,default=ROOT/'.env');p.add_argument('--run-id',default='reader-example-v1')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ['config','setup','apply-feedback']:sub.add_parser(name)
    s=sub.add_parser('seed');s.add_argument('--models',choices=['local','local,cloud'],default='local,cloud');s.add_argument('--enable-paid',action='store_true')
    d=sub.add_parser('demo');d.add_argument('--enable-paid',action='store_true')
    k=sub.add_parser('check');k.add_argument('--split',choices=['seed','demo'],required=True)
    i=sub.add_parser('inspect');i.add_argument('--message-id',required=True)
    return p
def main(argv=None):
    args=parser().parse_args(argv);config=load_config(args.env_file)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.run_id):raise ValueError('Invalid run ID')
    if args.command=='config':print(json.dumps(config.public(),indent=2));return
    paid=getattr(args,'enable_paid',False)
    if args.command in ['seed','demo'] and (args.command=='demo' or 'cloud' in args.models) and not paid:
        raise ValueError('Explicit --enable-paid required for possible cloud calls; no model request made')
    if paid and not config.openai_key:raise ValueError('Configure OPENAI_API_KEY privately; no request made')
    c=config.contract();work=ROOT/'artifacts'/args.run_id
    if args.command=='setup':
        from actian_vectorai import VectorAIClient
        with VectorAIClient(config.endpoint,api_key=config.api_key) as client:
            print(json.dumps(ensure_collection(client,config.endpoint,config.collection,config.api_key),indent=2))
        return
    bind(work,c,config)
    if args.command=='check':
        from .reader_checker import check_run
        print(json.dumps(check_run(work,c,args.split),indent=2));return
    store=ReaderStore(config)
    try:
        journal=Journal(work/'attempts.sqlite3')
        if args.command=='apply-feedback':result=apply_feedback(work,store,journal)
        elif args.command=='inspect':
            row=next(r for split in ['seed','demo'] for r in inputs(split) if r['id']==args.message_id)
            v=read(work/'vectors'/(row['id']+'.json'))['vector'];result=route(store,v,feedback(work),row['message'])
        else:
            if paid:os.environ['OPENAI_API_KEY']=config.openai_key
            ledger=Ledger(config.ledger)
            if ledger.summary()['ceiling_microusd']>750000:raise ValueError('Shared ledger cap must not exceed $0.75')
            local=LocalProvider(c,config.ollama_host);cloud=CloudProvider(c,ledger,paid)
            engine=Engine(c,local,cloud,journal,store,VERSION+':'+fingerprint(read(work/'configuration.json')))
            try:result=seed(work,c,engine,tuple(args.models.split(','))) if args.command=='seed' else demo(work,c,engine)
            finally:local.runtime.unload(local.runtime.generation);local.runtime.unload(local.runtime.embedding)
        print(json.dumps(result,indent=2))
    finally:store.close()
if __name__=='__main__':main()
