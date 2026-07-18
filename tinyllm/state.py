"""In-memory state: cooldown tracking and request metrics.

All state is kept in memory — no database.  Resets on restart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock

from .config import AppConfig


@dataclass
class ProviderState:
    """Cooldown & health state for a single provider+model combination."""

    consecutive_failures: int = 0
    last_error_time: float = 0.0
    cooldown_until: float = 0.0
    last_error_type: str = ""
    last_success_time: float = 0.0


@dataclass
class Metrics:
    """Aggregate request metrics (reset on restart)."""

    total_requests: int = 0
    successful_requests: int = 0
    total_fallbacks: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    total_latency_ms: float = 0.0


class AppState:
    """Thread-safe application state container."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._lock = Lock()
        self._provider_states: dict[str, ProviderState] = {}
        self.metrics = Metrics()
        self._start_time = time.time()

    # -- cooldown ----------------------------------------------------------

    @staticmethod
    def _key(provider: str, model: str) -> str:
        return f"{provider}\x00{model}"

    def is_cooldown_active(self, provider: str, model: str) -> bool:
        """Return True if *provider:model* is currently cooling down."""
        with self._lock:
            key = self._key(provider, model)
            state = self._provider_states.get(key)
            if not state:
                return False
            if time.time() < state.cooldown_until:
                return True
            # Cooldown expired — clean up and return False
            del self._provider_states[key]
            return False

    def mark_error(self, provider: str, model: str, error_type: str) -> None:
        """Record a failure for *provider:model* and start cooldown."""
        with self._lock:
            key = self._key(provider, model)
            state = self._provider_states.get(key)
            if state is None:
                state = ProviderState()
                self._provider_states[key] = state

            now = time.time()
            state.consecutive_failures += 1
            state.last_error_time = now
            state.last_error_type = error_type
            state.cooldown_until = now + self.config.cooldown_seconds

        self.metrics.errors_by_type[error_type] = (
            self.metrics.errors_by_type.get(error_type, 0) + 1
        )

    def mark_success(self, provider: str, model: str) -> None:
        """Record a successful response — clear any cooldown state."""
        with self._lock:
            key = self._key(provider, model)
            state = self._provider_states.get(key)
            if state:
                state.consecutive_failures = 0
                state.last_success_time = time.time()
                state.cooldown_until = 0.0

    # -- queries -----------------------------------------------------------

    def get_cooldown_summary(self) -> list[dict]:
        """Return list of currently cooling endpoints."""
        result: list[dict] = []
        now = time.time()
        with self._lock:
            for key, state in self._provider_states.items():
                remaining = state.cooldown_until - now
                if remaining > 0:
                    provider, model = key.split("\x00", 1)
                    result.append(
                        {
                            "provider": provider,
                            "model": model,
                            "remaining_seconds": round(remaining, 1),
                            "error_type": state.last_error_type,
                        }
                    )
        return result

    def get_metrics_snapshot(self) -> dict:
        """Return a snapshot of current metrics for diagnostics."""
        m = self.metrics
        uptime = time.time() - self._start_time
        avg_latency = (
            round(m.total_latency_ms / m.total_requests, 1)
            if m.total_requests
            else 0.0
        )
        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": m.total_requests,
            "successful_requests": m.successful_requests,
            "failed_requests": m.total_requests - m.successful_requests,
            "total_fallbacks": m.total_fallbacks,
            "errors_by_type": dict(m.errors_by_type),
            "average_latency_ms": avg_latency,
            "cooling_providers": len(self.get_cooldown_summary()),
        }
