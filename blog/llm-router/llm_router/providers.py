import os
import time
from .budget import BudgetStop, PRICE_NOTE
from .storage import vector_check
from local_runtime import LocalRuntime, OPTIONS, sustained_pressure, memory


class ProviderFailure(RuntimeError):
    def __init__(self, message, detail=None, fatal=False):
        super().__init__(message)
        self.detail = detail or {}
        self.fatal = fatal


class LocalProvider:
    def __init__(self, contract, host='http://localhost:11434'):
        self.contract = contract
        s = contract.settings
        self.options=s['options']
        self.runtime = LocalRuntime(host, s['local_model'], s['embedding_model'])
        tags = {m.model:m.digest for m in self.runtime.client.list().models}
        if tags.get(s['local_model']) != s['local_manifest_digest'] or tags.get(s['embedding_model']) != s['embedding_manifest_digest']:
            raise ValueError('Installed Ollama model digest differs from frozen settings')

    def embed(self, message):
        with self.runtime.prepared('embedding') as samples:
            response = self.runtime.client.embed(model=self.runtime.embedding, input=message, keep_alive='30s', options={'num_ctx':2048})
        if sustained_pressure(samples):
            raise MemoryError('Sustained severe RAM pressure during embedding; stop further inference')
        vector = response.embeddings[0]; vector_check(vector)
        return vector

    def generate(self, message, attempt_id):
        attempted = False; start = time.perf_counter()
        try:
            with self.runtime.prepared('generation') as samples:
                attempted = True
                response = self.runtime.client.chat(model=self.runtime.generation,
                    messages=[{'role':'system','content':self.contract.prompt},{'role':'user','content':message}],
                    format=self.contract.schema, think=False, options=self.options, keep_alive='30s')
                loaded = self.runtime.loaded()
            result = dict(text=response.message.content, raw_response=response.model_dump(mode='json'),
                          generation_attempted=True, provider_model=self.runtime.generation,
                          token_usage=dict(input_tokens=response.prompt_eval_count, output_tokens=response.eval_count),
                          actual_cost_microusd=0, price_assumption='Local inference; no cloud charge; electricity/hardware not costed.',
                          completion_ok=response.done is True and response.done_reason != 'length',
                          memory_samples=samples, memory_after_unload=memory(), loaded_models=loaded,
                          provider_latency_seconds=time.perf_counter()-start)
            if sustained_pressure(samples):
                raise ProviderFailure('Sustained severe RAM pressure; response preserved, stop further inference', result, fatal=True)
            return result
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(str(exc), dict(generation_attempted=attempted, memory=memory()), fatal=isinstance(exc,MemoryError)) from exc


class CloudProvider:
    def __init__(self, contract, ledger, enable_paid=False, sdk=None, model=None):
        self.contract, self.ledger, self.enable_paid, self.sdk = contract, ledger, enable_paid, sdk
        self.model = model or contract.settings['openai_model']

    def generate(self, message, attempt_id):
        # Explicit CLI flag only; env/config cannot quietly enable a paid call.
        if not self.enable_paid:
            raise ProviderFailure('Paid inference disabled; no OpenAI call made', dict(generation_attempted=False), fatal=True)
        if self.model not in ['gpt-5.4','gpt-5.4-2026-03-05']:
            raise ProviderFailure('Model lacks an approved versioned pricing envelope', fatal=True)
        if self.sdk is None:
            if not os.environ.get('OPENAI_API_KEY'):
                raise ProviderFailure('OPENAI_API_KEY not configured', fatal=True)
            from openai import OpenAI
            self.sdk = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url='https://api.openai.com/v1', max_retries=0, timeout=120)
        schema = dict(self.contract.schema); schema.pop('$id',None)
        payload = dict(model=self.model, instructions=self.contract.prompt, input=message,
                       text={'format':{'type':'json_schema','name':'support_extraction','schema':schema,'strict':True}},
                       reasoning={'effort':'none'}, max_output_tokens=256, store=False)
        reservation = self.ledger.reserve(attempt_id, payload, 256)
        raw = None; start = time.perf_counter()
        try:
            response = self.sdk.responses.create(**payload)
            raw = response.model_dump(mode='json')
            usage = response.usage
            if usage is None:
                raise BudgetStop('No trustworthy usage returned; reservation remains exposed')
            cost = self.ledger.settle(attempt_id, usage.input_tokens, usage.output_tokens)
            return dict(text=response.output_text, raw_response=raw, generation_attempted=True,
                        provider_model=response.model, token_usage=usage.model_dump(mode='json'), actual_cost_microusd=cost,
                        reservation=reservation, price_assumption=PRICE_NOTE, completion_ok=response.status=='completed',
                        provider_latency_seconds=time.perf_counter()-start)
        except Exception as exc:
            # Includes HTTP errors, timeouts, transport ambiguity, malformed usage and
            # process crashes (pending persists). No retries and no zero-cost assumption.
            self.ledger.uncertain(attempt_id, type(exc).__name__)
            raise ProviderFailure('OpenAI outcome cannot be safely reconciled; stop for billing review',
                                  dict(generation_attempted=True, raw_response=raw, reservation=reservation,
                                       price_assumption=PRICE_NOTE, billing_uncertain=True), fatal=True) from exc
