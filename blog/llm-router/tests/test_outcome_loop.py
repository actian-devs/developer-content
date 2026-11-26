import unittest
from unittest.mock import Mock
from unittest.mock import patch
from llm_router.outcome_loop import choose,OutcomeStore,make_feedback

def hit(i,passed,kind='local',key=None):
 return dict(id=str(i),score=.8,payload=dict(model_kind=kind,input_hash=key or str(i),passed_check=passed))
class OutcomeTests(unittest.TestCase):
 def test_failures_stay_in_denominator_despite_success_filter(self):
  allhits=[hit(i,i<3) for i in range(5)];success=allhits[:3]
  d=choose(allhits,success,'query');self.assertEqual(d['selected_model'],'cloud')
  self.assertEqual(d['tiers']['local']['match_rate'],.6)
  d=choose([hit(i,i<4) for i in range(5)],[hit(i,True) for i in range(4)],'query')
  self.assertEqual(d['selected_model'],'local')
 def test_distinct_and_model_specific_evidence(self):
  a=[hit(i,True,key='same') for i in range(8)]+[hit('cloud'+str(i),True,'cloud') for i in range(6)]
  d=choose(a,a,'query');self.assertEqual(d['tiers']['local']['distinct_support'],1);self.assertEqual(d['selected_model'],'cloud')
  a=[hit(i,True,key='same') for i in range(5)]+[hit('failure',False,key='same')]
  self.assertEqual(choose(a,a[:5],'query')['tiers']['local']['failed'],1)
 def test_five_minimum_and_self_exclusion(self):
  a=[hit(i,True) for i in range(4)];self.assertEqual(choose(a,a,'query')['selected_model'],'cloud')
  a.append(hit(4,True,key='query'));self.assertEqual(choose(a,a,'query')['selected_model'],'cloud')
 def test_native_success_query_has_both_must_conditions(self):
  s=object.__new__(OutcomeStore);s.client=Mock()
  s.client.collections.get_info.return_value.model_dump.return_value={'points_count':600}
  s.client.points.search.return_value=[Mock(model_dump=Mock(return_value={'id':'p','payload':{'passed_check':True},'score':.8}))]
  records=[{'feedback_id':'p','passed_check':True},{'feedback_id':'f','passed_check':False}]
  s.search_successes([1.]+[0.]*767,records)
  f=s.client.points.search.call_args.kwargs['filter'].to_proto()
  self.assertEqual(len(f.must),2);self.assertEqual(f.must[1].field.key,'passed_check');self.assertTrue(f.must[1].field.match.boolean)
  params=s.client.points.search.call_args.kwargs['params'].to_proto()
  self.assertFalse(params.HasField('exact'));self.assertEqual(params.hnsw_ef,4096)
 def test_feedback_rejects_evaluation_and_runtime_only_checks(self):
  a=dict(message_id='E-001',message='x',stage='evaluation')
  with self.assertRaises(ValueError):make_feedback(a,True,{'kind':'human_review','evidence_sha256':'fixture'},{'E-001':'x'})
  from llm_router.contract import message_key
  a=dict(message_id='D-001',message='x',stage='demo')
  with self.assertRaises(ValueError):make_feedback(a,True,{'kind':'runtime_validation','evidence_sha256':'fixture'},{'D-001':message_key('x')})

 def test_explicit_semantic_rejection_escalates_and_writes_both_bound_outcomes(self):
  from llm_router.outcome_loop import execute
  from llm_router.contract import fingerprint
  engine=Mock();store=Mock()
  engine.attempt.side_effect=[dict(attempt_id='local-fixture',runtime_check_pass=True),dict(attempt_id='cloud-fixture',runtime_check_pass=True)]
  def checker(a):return dict(source_attempt_id=a['attempt_id'],source_attempt_hash=fingerprint(a),passed_check=a['attempt_id']=='cloud-fixture')
  with patch('llm_router.outcome_loop.route',return_value={'selected_model':'local','reason':'SIMULATED_SELECTION'}):
   r=execute(engine,store,[1.]+[0.]*767,[],'fixture','D-fixture','SIMULATED',checker)
  self.assertTrue(r['escalated']);self.assertEqual(r['attempt_ids'],['local-fixture','cloud-fixture'])
  self.assertEqual(store.upsert_feedback.call_count,2)
  self.assertEqual(len({call.args[2] for call in engine.attempt.call_args_list}),1)

 def test_no_semantic_feedback_does_not_create_success(self):
  from llm_router.outcome_loop import execute
  engine=Mock();store=Mock();engine.attempt.return_value=dict(attempt_id='cloud-fixture',runtime_check_pass=True)
  with patch('llm_router.outcome_loop.route',return_value={'selected_model':'cloud'}):
   r=execute(engine,store,[1.]+[0.]*767,[],'fixture','D-fixture','SIMULATED')
  store.upsert_feedback.assert_not_called();self.assertEqual(r['feedback_records'],[])

 def test_future_live_feedback_requires_explicit_review_and_exact_allowlist(self):
  from llm_router.contract import message_key
  a=dict(message_id='live-fixture',message='x',stage='live',attempt_id='fixture',model_kind='cloud',runtime_check_pass=True,
      actual_cost_microusd=0,local_manifest_digest='fixture',prompt_sha256='fixture')
  allow={'live-fixture':message_key('x')}
  with self.assertRaises(ValueError):make_feedback(a,True,{'kind':'synthetic_draft_reference_exact_match','evidence_sha256':'fixture'},allow)
  f=make_feedback(a,False,{'kind':'human_review','evidence_sha256':'fixture'},allow)
  self.assertEqual(f['semantic_status'],'reviewed_mismatch');self.assertFalse(f['passed_check'])
  with self.assertRaises(ValueError):make_feedback(a,True,{'kind':'human_review','evidence_sha256':'fixture'},{'live-fixture':'wrong'})
