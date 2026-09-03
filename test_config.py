import os
import unittest
from unittest.mock import patch

from tinyllm.config import AppConfig, ConfigError, load_config


TEST_ENV = {
    "TINYLLM_API_KEYS": "test-key",
    "OPENCODE_ZEN_API_KEY": "test-opencode",
    "OPENROUTER_API_KEY": "test-openrouter",
    "ZAI_API_KEY": "test-zai",
    "ORCAROUTER_API_KEY": "test-orca",
}

EXPECTED_ROUTES = {
    "agent-auto": [
        ("openrouter", "poolside/laguna-s-2.1:free"),
        ("opencode-zen", "laguna-s-2.1-free"),
        ("orcarouter", "deepseek/deepseek-v4-flash-free"),
    ],
    "coding-auto": [
        ("orcarouter", "deepseek/deepseek-v4-flash-free"),
        ("openrouter", "cohere/north-mini-code:free"),
        ("openrouter", "poolside/laguna-s-2.1:free"),
        ("opencode-zen", "laguna-s-2.1-free"),
    ],
    "agent-auto-pay": [
        ("openrouter", "poolside/laguna-s-2.1:free"),
        ("opencode-zen", "laguna-s-2.1-free"),
        ("orcarouter", "deepseek/deepseek-v4-flash-free"),
    ],
    "coding-deepseek": [
        ("orcarouter", "deepseek/deepseek-v4-flash-free"),
    ],
    "coding-auto-pay": [
        ("orcarouter", "deepseek/deepseek-v4-flash-free"),
        ("openrouter", "cohere/north-mini-code:free"),
        ("openrouter", "poolside/laguna-s-2.1:free"),
        ("opencode-zen", "laguna-s-2.1-free"),
        ("openrouter", "deepseek/deepseek-v4-flash-0731"),
    ],
}


class ConfigTests(unittest.TestCase):
    def load_current_config(self):
        with patch.dict(os.environ, TEST_ENV, clear=False):
            return load_config("config.yaml")

    def test_current_config_loads_provider_headers(self):
        config = self.load_current_config()

        self.assertNotIn("headers", config.providers)
        self.assertEqual(
            config.providers["opencode-zen"].headers,
            {
                "User-Agent": "OpenAI/JS 6.45.0",
            },
        )
        self.assertEqual(
            config.providers["openrouter"].headers,
            {
                "HTTP-Referer": "https://llm.stpnk.tech",
                "X-Title": "TinyLLM",
            },
        )

    def test_current_config_has_expected_routes(self):
        config = self.load_current_config()

        self.assertEqual(set(config.route_names), set(EXPECTED_ROUTES))
        for route_name, expected_steps in EXPECTED_ROUTES.items():
            actual_steps = [
                (step.provider, step.model)
                for step in config.routes[route_name].steps
            ]
            self.assertEqual(actual_steps, expected_steps)

    def test_current_config_routes_reference_known_providers(self):
        config = self.load_current_config()

        for route in config.routes.values():
            for step in route.steps:
                self.assertIn(step.provider, config.providers)

    def test_max_attempts_covers_current_config_routes(self):
        config = self.load_current_config()

        longest_route = max(len(route.steps) for route in config.routes.values())

        self.assertGreaterEqual(config.max_attempts, longest_route)

    def test_route_steps_to_try_respects_max_attempts(self):
        from tinyllm import handlers

        data = {
            "auth": {"api_keys_env": "TINYLLM_API_KEYS"},
            "routing": {"max_attempts": 2},
            "providers": {
                "openrouter": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key_env": "OPENROUTER_API_KEY",
                }
            },
            "routes": {
                "limited": [
                    {"provider": "openrouter", "model": "model-a"},
                    {"provider": "openrouter", "model": "model-b"},
                    {"provider": "openrouter", "model": "model-c"},
                ]
            },
        }

        with patch.dict(os.environ, TEST_ENV, clear=False):
            config = AppConfig(data)

        steps = handlers._route_steps_to_try(config.routes["limited"], config)

        self.assertEqual([step.model for step in steps], ["model-a", "model-b"])

    def test_config_rejects_routes_with_unknown_providers(self):
        data = {
            "auth": {"api_keys_env": "TINYLLM_API_KEYS"},
            "providers": {
                "openrouter": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key_env": "OPENROUTER_API_KEY",
                }
            },
            "routes": {
                "broken": [
                    {"provider": "z-ai", "model": "z-ai/glm-5.2:free"},
                ]
            },
        }

        with patch.dict(os.environ, TEST_ENV, clear=False):
            with self.assertRaisesRegex(ConfigError, "unknown provider"):
                AppConfig(data)


if __name__ == "__main__":
    unittest.main()
