"""The endpoint. Cache check first, model second, write back on the way out."""

import time

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from app.cache import SemanticCache
from app.llm import complete

app = FastAPI()
cache = SemanticCache()


class Ask(BaseModel):
    question: str
    scope: str | None = None


class Answer(BaseModel):
    answer: str
    cached: bool
    ms: float


@app.post("/ask", response_model=Answer)
def ask(body: Ask, background: BackgroundTasks):
    started = time.perf_counter()

    hit = cache.lookup(body.question, scope=body.scope)
    if hit:
        return Answer(answer=hit["answer"], cached=True, ms=(time.perf_counter() - started) * 1000)

    answer = complete(body.question)
    background.add_task(cache.store, body.question, answer, body.scope)
    return Answer(answer=answer, cached=False, ms=(time.perf_counter() - started) * 1000)
