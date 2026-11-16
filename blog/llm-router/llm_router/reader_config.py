"""Public reader configuration; never prints credentials or resets a ledger."""
import os,re
from dataclasses import dataclass
from pathlib import Path
from dotenv import dotenv_values
from .contract import ROOT,Contract

@dataclass
class Config:
    endpoint:str='localhost:6674'
    collection:str='vectorai_llm_router_tutorial_attempts_v1'
    ollama_host:str='http://localhost:11435'
    profile:str='qwen-memory-v2.json'
    budget:float=.75
    ledger:Path=ROOT/'runs/experiment-budget.sqlite3'
    api_key:str|None=None
    openai_key:str|None=None
    def public(self):
        return dict(endpoint=self.endpoint,collection=self.collection,ollama_host=self.ollama_host,profile=self.profile,
            budget=self.budget,ledger=str(self.ledger),paid_enabled_by_default=False,credentials_configured=bool(self.openai_key))
    def contract(self):
        previous=os.environ.get('ROUTER_INFERENCE_PROFILE');os.environ['ROUTER_INFERENCE_PROFILE']=self.profile
        try:return Contract()
        finally:
            if previous is None:os.environ.pop('ROUTER_INFERENCE_PROFILE',None)
            else:os.environ['ROUTER_INFERENCE_PROFILE']=previous

def load_config(env_file=ROOT/'.env',environ=None):
    values={**(dotenv_values(env_file) if env_file and Path(env_file).exists() else {}),**(os.environ if environ is None else environ)}
    c=Config(endpoint=values.get('ACTIAN_VECTORAI_URL','localhost:6674'),collection=values.get('ACTIAN_COLLECTION','vectorai_llm_router_tutorial_attempts_v1'),
        ollama_host=values.get('OLLAMA_HOST','http://localhost:11435'),profile=values.get('ROUTER_INFERENCE_PROFILE','qwen-memory-v2.json'),
        budget=float(values.get('EXPERIMENT_BUDGET_USD','.75')),api_key=values.get('ACTIAN_VECTORAI_API_KEY') or None,openai_key=values.get('OPENAI_API_KEY') or None)
    if not re.fullmatch(r'(localhost|127\.0\.0\.1):\d+',c.endpoint):raise ValueError('Reader path supports local plaintext gRPC only; configure TLS separately before using a remote DB')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',c.collection):raise ValueError('Invalid collection name')
    if c.profile!='qwen-memory-v2.json' or c.ollama_host!='http://localhost:11435' or c.budget!=.75:raise ValueError('Use qwen-memory-v2, Ollama port 11435, and the unchanged $0.75 ceiling')
    if values.get('EXPERIMENT_LEDGER','runs/experiment-budget.sqlite3').replace('\\','/')!='runs/experiment-budget.sqlite3':raise ValueError('Use the existing shared experiment ledger; no alternate budget path')
    contract=c.contract()
    for key,expected in [('LOCAL_MODEL',contract.settings['local_model']),('EMBEDDING_MODEL',contract.settings['embedding_model']),('OPENAI_MODEL',contract.settings['openai_model'])]:
        if key in values and values[key]!=expected:raise ValueError('Public model setting conflicts with frozen profile: '+key)
    return c
