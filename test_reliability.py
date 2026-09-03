import threading
import unittest
from types import SimpleNamespace

from tinyllm import handlers
from tinyllm.reliability import UpstreamRegistry


def step(provider, model):
    return SimpleNamespace(provider=provider, model=model)


class ReliabilityTests(unittest.TestCase):
    def test_snapshot_does_not_deadlock_when_latency_exists(self):
        registry = UpstreamRegistry()
        stats = registry.get_or_create("openrouter", "model-a")
        stats.record(status=200, empty=False, timeout=False, latency_ms=123.0)

        result = []
        worker = threading.Thread(target=lambda: result.append(registry.snapshot()), daemon=True)
        worker.start()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result[0][0]["p95_ms"], 123.0)

    def test_trust_filter_deprioritizes_untrusted_steps(self):
        registry = UpstreamRegistry()
        bad = registry.get_or_create("orcarouter", "deepseek")
        for _ in range(20):
            bad.record(status=500, empty=False, timeout=False, latency_ms=50.0)

        ordered, dropped = registry.trust_filter(
            [step("orcarouter", "deepseek"), step("openrouter", "poolside")],
            min_requests=20,
            min_success_rate=0.5,
            max_empty_rate=0.3,
            min_score=0.0,
        )

        self.assertEqual([s.model for s in ordered], ["poolside", "deepseek"])
        self.assertEqual(dropped[0]["provider"], "orcarouter")

    def test_cooldown_filter_skips_cooling_steps_when_others_exist(self):
        route = SimpleNamespace(
            name="agent-auto",
            steps=[step("openrouter", "poolside"), step("openrouter", "fallback")],
        )
        config = SimpleNamespace(
            max_attempts=5,
            min_requests_for_trust=20,
            min_success_rate=0.5,
            max_empty_rate=0.3,
            min_score=0.0,
        )
        state = SimpleNamespace(
            upstream_registry=UpstreamRegistry(),
            is_cooldown_active=lambda provider, model: model == "poolside",
        )

        ordered = handlers._route_steps_to_try_with_cooldown(
            route, config, state, "req-test"
        )

        self.assertEqual([s.model for s in ordered], ["fallback"])

    def test_cooldown_filter_keeps_route_when_all_steps_are_cooling(self):
        route = SimpleNamespace(
            name="agent-auto",
            steps=[step("openrouter", "poolside"), step("openrouter", "fallback")],
        )
        config = SimpleNamespace(
            max_attempts=5,
            min_requests_for_trust=20,
            min_success_rate=0.5,
            max_empty_rate=0.3,
            min_score=0.0,
        )
        state = SimpleNamespace(
            upstream_registry=UpstreamRegistry(),
            is_cooldown_active=lambda provider, model: True,
        )

        ordered = handlers._route_steps_to_try_with_cooldown(
            route, config, state, "req-test"
        )

        self.assertEqual([s.model for s in ordered], ["poolside", "fallback"])


if __name__ == "__main__":
    unittest.main()
