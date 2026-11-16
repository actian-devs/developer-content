"""Throughput / ETA logging."""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class Progress:
    def __init__(self, total: int, *, label: str = "migrated", interval: float = 10.0, done: int = 0) -> None:
        self.total = total
        self.label = label
        self.interval = interval
        self.done = done
        self._start_done = done
        self._t0 = time.monotonic()
        self._last = self._t0

    def advance(self, n: int, *, force: bool = False) -> None:
        self.done += n
        now = time.monotonic()
        if force or now - self._last >= self.interval:
            self._last = now
            elapsed = max(now - self._t0, 1e-9)
            rate = (self.done - self._start_done) / elapsed
            remaining = max(self.total - self.done, 0)
            eta = f"{remaining / rate:,.0f}s" if rate > 0 else "n/a"
            pct = 100.0 * self.done / self.total if self.total else 100.0
            log.info("%s %d/%d (%.1f%%) | %.0f rec/s | ETA %s", self.label, self.done, self.total, pct, rate, eta)
