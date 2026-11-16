"""Configured collection adapter, reusing native retrieval and outcome-loop code."""
from actian_vectorai import VectorAIClient,PointStruct
from .outcome_loop import OutcomeStore
from .contract import fingerprint,assert_runtime_only
from .storage import vector_check
from .collection_setup import read_config,compatibility

class ReaderStore(OutcomeStore):
    def __init__(self,config):
        self.collection=config.collection;self.endpoint=config.endpoint;self.search_audit=[]
        self.client=VectorAIClient(config.endpoint,api_key=config.api_key)
        self.client.connect()
        try:compatibility(read_config(config.endpoint,self.collection,config.api_key))
        except Exception:
            self.client.close();raise
        self.client.vde.open_collection(self.collection)
        self.collection_info=self.client.collections.get_info(self.collection).model_dump(mode='json')
    def read(self,ids):
        if not ids:return []
        return [p.model_dump(mode='json') for p in self.client.points.get(self.collection,ids,with_payload=True,timeout=15)]
    def write(self,record,vector):
        assert_runtime_only(record);vector_check(vector)
        if self.read([record['attempt_id']]):raise ValueError('Refusing overwrite of existing attempt')
        self.client.points.upsert(self.collection,[PointStruct(id=record['attempt_id'],vector=vector,payload=record)],timeout=15)
        got=self.read([record['attempt_id']])
        if len(got)!=1 or fingerprint(got[0]['payload'])!=fingerprint(record):raise RuntimeError('Attempt readback failed; durable journal retained')
