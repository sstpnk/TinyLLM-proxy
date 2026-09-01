"""Per-upstream reliability tracking.

Each ``(provider, model)`` pair keeps:
 - ``requests_total``
 - ``requests_empty`` (HTTP 200 but no content / no reasoning_content)
 - ``requests_error_4xx`` / ``requests_error_5xx`` / ``requests_timeout``
 - latency p95 over a sliding window of the last ``WINDOW_SIZE`` samples

A composite ``score`` summarises reliability; ``handler._route_steps_to_try``
uses it to reorder and possibly drop low-trust models.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

WINDOW_SIZE = 100


@dataclass
class UpstreamStats:
    """Reliability counters for a single (provider, model)."""

    requests_total: int = 0
    requests_empty: int = 0
    requests_error_4xx: int = 0
    requests_error_5xx: int = 0
    requests_timeout: int = 0
    _latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_update: float = 0.0

    def record(
        self,
        *,
        status: int | None,
        empty: bool,
        timeout: bool,
        latency_ms: float,
    ) -> None:
        now = time.time()
        with self._lock:
            self.requests_total += 1
            self._last_update = now
            if timeout:
                self.requests_timeout += 1
            elif status is not None and 400 <= status < 500:
                self.requests_error_4xx += 1
            elif status is not None and status >= 500:
                self.requests_error_5xx += 1
            elif empty:
                self.requests_empty += 1
            self._latencies_ms.append(latency_ms)

    @property
    def latency_p95_ms(self) -> float:
        with self._lock:
            if not self._latencies_ms:
                return 0.0
            sample = sorted(self._latencies_ms)
            idx = max(0, math.ceil(0.95 * len(sample)) - 1)
            return sample[idx]

    def score(self, *, min_requests: int) -> float:
        """Composite reliability score.

        Returns ``None`` (encoded as ``-inf`` via :func:`score_for`) when there
        aren't enough samples yet — caller decides what to do with that.
        """
        with self._lock:
            n = self.requests_total
            if n < min_requests:
                return float("-inf")
            failures = (
                self.requests_error_4xx
                + self.requests_error_5xx
                + self.requests_timeout
            )
            success_rate = (n - failures) / n
            empty_rate = self.requests_empty / n
        p95 = self.latency_p95_ms
        return success_rate * 100.0 - 5.0 * empty_rate * 100.0 - 2.0 * (p95 / 1000.0)


class UpstreamRegistry:
    """Thread-safe collection of :class:`UpstreamStats` keyed by ``(provider, model)``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[tuple[str, str], UpstreamStats] = {}

    def key(self, provider: str, model: str) -> tuple[str, str]:
        return (provider, model)

    def get_or_create(self, provider: str, model: str) -> UpstreamStats:
        with self._lock:
            k = self.key(provider, model)
            item = self._items.get(k)
            if item is None:
                item = UpstreamStats()
                self._items[k] = item
            return item

    def snapshot(self) -> list[dict]:
        with self._lock:
            items = list(self._items.items())
        out: list[dict] = []
        for (provider, model), s in items:
            with s._lock:
                out.append(
                    {
                        "provider": provider,
                        "model": model,
                        "requests": s.requests_total,
                        "empty": s.requests_empty,
                        "errors_4xx": s.requests_error_4xx,
                        "errors_5xx": s.requests_error_5xx,
                        "timeouts": s.requests_timeout,
                        "p95_ms": s.latency_p95_ms,
                        "last_update": s._last_update,
                    }
                )
        return out

    def trust_filter(
        self,
        steps: list,
        *,
        min_requests: int,
        min_success_rate: float,
        max_empty_rate: float,
        min_score: float,
    ) -> tuple[list, list[dict]]:
        """Reorder *steps* by score and drop untrusted ones.

        Returns ``(filtered_steps, dropped)`` where ``dropped`` is a list of
        ``{"provider": ..., "model": ..., "reason": ...}`` for logging.
        """
        scored: list[tuple[float, object]] = []
        dropped: list[dict] = []
        for step in steps:
            stats = self.get_or_create(step.provider, step.model)
            sc = stats.score(min_requests=min_requests)
            n = stats.requests_total
            empty_rate = stats.requests_empty / n if n else 0.0
            success_rate = (
                (n - stats.requests_error_4xx - stats.requests_error_5xx - stats.requests_timeout)
                / n
                if n
                else 0.0
            )

            reason: str | None = None
            if n >= min_requests and success_rate < min_success_rate:
                reason = f"success_rate<{min_success_rate:.2f}"
            elif empty_rate > max_empty_rate:
                reason = f"empty_rate>{max_empty_rate:.2f}"
            elif sc != float("-inf") and sc < min_score:
                reason = f"score<{min_score:.1f}"

            if reason is not None:
                dropped.append(
                    {
                        "provider": step.provider,
                        "model": step.model,
                        "reason": reason,
                        "score": sc,
                    }
                )
                continue
            scored.append((sc, step))

        # Stable sort: models without score (sc=-inf) keep original order,
        # models with score go highest first.
        indexed = [(idx, sc, step) for idx, (sc, step) in enumerate(scored)]
        indexed.sort(key=lambda t: (-float("-inf") if t[1] == float("-inf") else -t[1], t[0]))
        return [step for _, _, step in indexed], dropped
