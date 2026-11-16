"""Small retry helper with capped exponential backoff and jitter."""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0, rng: Callable[[], float] = random.random) -> float:
    """Delay before retry number ``attempt`` (1-based): min(cap, base*2^(n-1)) with 50-100% jitter."""
    return min(cap, base * (2 ** (attempt - 1))) * (0.5 + rng() / 2)


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int,
    is_retryable: Callable[[BaseException], bool],
    what: str,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    delay_for: Callable[[BaseException, int], float] | None = None,
) -> T:
    """Call ``fn`` up to ``attempts`` times; re-raise the last error if it is not retryable or attempts run out."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classification is delegated to is_retryable
            if attempt >= attempts or not is_retryable(exc):
                raise
            delay = delay_for(exc, attempt) if delay_for else backoff_delay(attempt, base_delay, max_delay)
            log.warning("%s failed (%s: %s); retry %d/%d in %.1fs", what, type(exc).__name__, exc, attempt, attempts - 1, delay)
            sleep(delay)
