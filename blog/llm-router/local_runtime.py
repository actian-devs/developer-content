"""Sequential Ollama access with model switching and memory safety gates."""
import threading
import time
from contextlib import contextmanager
import ollama
import psutil
from llm_router.runlock import ProjectInferenceLock

OPTIONS = dict(num_ctx=2048, num_predict=256, temperature=0, seed=42, num_thread=6)
MIN_START = int(1.5 * 1024**3)
SEVERE = 512 * 1024**2


def memory():
    m = psutil.virtual_memory()
    return dict(available_bytes=m.available, total_bytes=m.total)


class LocalRuntime:
    def __init__(self, host='http://localhost:11434', generation='qwen3:4b-instruct', embedding='embeddinggemma'):
        self.client = ollama.Client(host=host, timeout=120)
        self.generation, self.embedding = generation, embedding
        self.lock = threading.Lock()
        self.events = []

    def loaded(self):
        return self.client.ps().model_dump(mode='json')

    def unload(self, model):
        names = [m['name'] for m in self.loaded()['models']]
        target = model if ':' in model else model + ':latest'
        if target not in names:
            return
        # Empty generate with keep_alive=0 is Ollama's unload mechanism, including embed models.
        self.client.generate(model=model, keep_alive=0)
        for _ in range(20):
            if target not in [m['name'] for m in self.loaded()['models']]:
                self.events.append(dict(event='unloaded', model=model, memory=memory()))
                return
            time.sleep(.25)
        raise RuntimeError('Model did not unload; refusing next inference')

    @contextmanager
    def prepared(self, kind):
        with self.lock, ProjectInferenceLock():
            self.unload(self.embedding if kind == 'generation' else self.generation)
            # Do not displace an unrelated model to obtain RAM.
            allowed = {self.generation, self.embedding if ':' in self.embedding else self.embedding+':latest'}
            if any(m['name'] not in allowed for m in self.loaded()['models']):
                raise RuntimeError('An unrelated Ollama model is loaded; leave it untouched')
            before = memory()
            self.events.append(dict(event='before_'+kind, memory=before, loaded=self.loaded()))
            if before['available_bytes'] < MIN_START:
                raise MemoryError('Less than 1.5 GiB available after unloading; free memory before inference')
            samples, stop = [], threading.Event()
            def sample():
                while not stop.is_set():
                    samples.append(dict(t=time.perf_counter(), **memory()))
                    stop.wait(.5)
            worker = threading.Thread(target=sample, daemon=True); worker.start()
            try:
                yield samples
            finally:
                stop.set(); worker.join()
                self.unload(self.generation if kind == 'generation' else self.embedding)
                self.events.append(dict(event='after_'+kind, memory=memory(), loaded=self.loaded()))


def sustained_pressure(samples):
    start = None
    for s in samples:
        if s['available_bytes'] < SEVERE:
            if start is None: start = s['t']
            if s['t'] - start >= 5: return True
        else: start = None
    return False
