"""SIMULATED provider/DB fixtures. Never contact Ollama, Actian or OpenAI."""
import unittest,tempfile,json,re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from llm_router.reader_config import load_config
from llm_router.reader_store import ReaderStore
from llm_router import reader_tutorial as app
from llm_router import reader_checker
from llm_router.collection_setup import ensure_collection,compatibility
from llm_router.contract import ROOT
from llm_router.storage import Journal
from llm_router.engine import Engine

class Row:
 def __init__(self,data):self.data=data
 def model_dump(self,**kwargs):return self.data
class FixturePoints:
 def __init__(self):self.data={}
 def upsert(self,name,points,**kw):
  for p in points:self.data[str(p.id)]={'id':str(p.id),'payload':p.payload}
 def get(self,name,ids,**kw):return [Row(self.data[str(i)]) for i in ids if str(i) in self.data]
 def search(self,name,vector,filter,**kw):
  conditions=filter.to_proto().must;ids=[p.uuid for p in conditions[0].has_id.has_id]
  success=len(conditions)>1
  return [Row(dict(self.data[i],score=.8)) for i in ids if i in self.data and (not success or self.data[i]['payload']['passed_check'])]
class FixtureStore(ReaderStore):
 def __init__(self):
  self.collection='fixture';self.search_audit=[];points=FixturePoints()
  self.client=SimpleNamespace(points=points,collections=SimpleNamespace(get_info=lambda name:Row({'points_count':len(points.data)})))
class FixtureProvider:
 enable_paid=False
 def __init__(self,kind):self.kind=kind;self.calls=0
 def embed(self,message):return [1.]+[0.]*767
 def generate(self,message,aid):
  self.calls+=1;order=re.search(r'ORD-\d+',message).group()
  if self.kind=='local' and order=='ORD-123456':order='ORD-999999'
  text=json.dumps(dict(category='refund',order_id=order,requested_action='refund',urgency='normal',needs_clarification=False))
  return dict(text=text,raw_response={'SIMULATED':True},generation_attempted=False,completion_ok=True,actual_cost_microusd=0)

class ReaderTests(unittest.TestCase):
 def test_config_example_and_parser_defaults(self):
  cfg=load_config(ROOT/'.env.example',environ={});self.assertEqual(cfg.budget,.75);self.assertEqual(cfg.ollama_host,'http://localhost:11435')
  self.assertIsNone(cfg.openai_key);self.assertEqual(app.parser().parse_args(['seed']).models,'local,cloud')
  self.assertFalse(app.parser().parse_args(['demo']).enable_paid)
  with self.assertRaises(ValueError):load_config(None,{'EXPERIMENT_LEDGER':'new-ledger.sqlite3'})
  with self.assertRaises(ValueError):load_config(None,{'EXPERIMENT_BUDGET_USD':'3'})
 def test_paid_disabled_before_provider_or_db_construction(self):
  with patch.object(app,'LocalProvider') as provider,patch.object(app,'ReaderStore') as store:
   with self.assertRaises(ValueError):app.main(['--env-file',str(ROOT/'.env.example'),'seed'])
   provider.assert_not_called();store.assert_not_called()
 def test_idempotent_setup_and_incompatible_existing_collection(self):
  c=Mock();c.collections.exists.return_value=False
  cfg=lambda *args:{'params':{'vectors_config':{'params':{'size':'768','distance':'Cosine'}}}}
  self.assertTrue(ensure_collection(c,'localhost:1','fixture',config_reader=cfg)['created'])
  c.collections.exists.return_value=True
  self.assertFalse(ensure_collection(c,'localhost:1','fixture',config_reader=cfg)['created'])
  self.assertEqual(c.collections.create.call_count,1);c.collections.delete.assert_not_called()
  with self.assertRaises(ValueError):ensure_collection(c,'localhost:1','fixture',config_reader=lambda *a:{})
  with self.assertRaises(ValueError):compatibility({'params':{'vectors_config':{'params':{'size':384,'distance':'Cosine'}}}})
  with self.assertRaises(ValueError):compatibility({'params':{'vectors_config':{'params':{'size':768,'distance':'Euclid'}}}})
 def test_fresh_seed_check_writeback_demo_without_historical_artifacts(self):
  with tempfile.TemporaryDirectory() as directory:
   base=Path(directory);data=base/'data';work=base/'fresh'
   seed=[{'id':'S-001','message':'Please refund order ORD-123456.'},{'id':'S-002','message':'Please refund order ORD-123457.'}]
   demo=[{'id':'D-001','message':'Please refund order ORD-123458.'}]
   for split,rows in [('seed',seed),('demo',demo)]:
    app.save(data/'inputs'/(split+'.json'),rows)
    app.save(data/'offline_labels'/(split+'.json'),[dict(id=r['id'],reference=dict(category='refund',order_id=re.search(r'ORD-\d+',r['message']).group(),requested_action='refund',urgency='normal',needs_clarification=False)) for r in rows])
   cfg=load_config(None,environ={});c=cfg.contract();store=FixtureStore();local=FixtureProvider('local');cloud=FixtureProvider('cloud')
   j=Journal(work/'attempts.sqlite3');engine=Engine(c,local,cloud,j,store,'SIMULATED_READER_RUN')
   with patch.object(app,'DATA',data),patch.object(reader_checker,'DATA',data):
    app.bind(work,c,cfg);self.assertEqual(app.seed(work,c,engine,('local','cloud'))['completed_attempts'],4)
    app.seed(work,c,engine,('local','cloud'));self.assertEqual((local.calls,cloud.calls),(2,2))
    self.assertEqual(reader_checker.check_run(work,c,'seed')['checked'],4)
    app.apply_feedback(work,store,j);self.assertEqual(len(app.feedback(work)),4)
    self.assertTrue(any(not f['passed_check'] for f in app.feedback(work)))
    result=app.demo(work,c,engine);self.assertEqual(result['completed'],1)
    self.assertEqual(result['results'][0]['decision']['selected_model'],'cloud')
    self.assertEqual(result['results'][0]['final']['semantic_outcome'],'unverified')
    app.demo(work,c,engine);self.assertEqual(cloud.calls,3)
    reader_checker.check_run(work,c,'demo');app.apply_feedback(work,store,j)
    self.assertEqual(len(app.feedback(work)),5)
    self.assertEqual(len(j.all()),5)
 def test_runtime_has_no_label_file_dependency(self):
  import ast
  tree=ast.parse((ROOT/'llm_router/reader_tutorial.py').read_text())
  # Offline checker is loaded only in the explicitly selected check branch.
  imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
  self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module=='reader_checker' for n in imports))
  for file in ['outcome_loop.py','providers.py','reader_store.py']:
   self.assertNotIn('offline_labels',(ROOT/'llm_router'/file).read_text())
if __name__=='__main__':unittest.main()
