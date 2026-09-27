import os
import tempfile
import textwrap
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tinyllm import app as tinyllm_app
from tinyllm import handlers
from tinyllm.config import ConfigError, _extract_dynamic_yaml, load_config_with_dynamic


TEST_ENV = {
    "TINYLLM_API_KEYS": "test-key",
    "TINYLLM_ADMIN_TOKEN": "test-admin",
}


BASE_CONFIG = """
server:
  host: 127.0.0.1
  port: 4999
auth:
  api_keys_env: TINYLLM_API_KEYS
admin:
  token_env: TINYLLM_ADMIN_TOKEN
routing:
  cooldown_seconds: 60
  max_attempts: 1
timeouts:
  connect_seconds: 5
  response_seconds: 10
  stream_idle_seconds: 15
providers:
  openrouter:
    type: openai-compatible
    base_url: https://static-openrouter.example/v1
    api_key_env: OPENROUTER_API_KEY
routes:
  agent-auto:
    - provider: openrouter
      model: static-agent-model
"""


DYNAMIC_CONFIG = """
metadata:
  schema_version: 1
  generated_by: free-ai-model-router
providers:
  openrouter:
    type: openai-compatible
    base_url: https://dynamic-openrouter.example/v1
    api_key_env: OPENROUTER_API_KEY
  deepseek2api:
    type: openai-compatible
    base_url: https://deepseek.stpnk.tech/v1
    api_key_env: DEEPSEEK2API_API_KEY
routing:
  cooldown_seconds: 300
  max_attempts: 2
timeouts:
  connect_seconds: 30
routes:
  agent-auto:
    - provider: openrouter
      model: poolside/laguna-s-2.1:free
  coding-auto-generated:
    - provider: openrouter
      model: poolside/laguna-s-2.1:free
    - provider: deepseek2api
      model: deepseek-v4-flash
    - provider: deepseek2api
      model: deepseek-v4-pro
  empty-route: []
"""


class DynamicConfigTests(unittest.TestCase):
    def write_config(self, directory: str, name: str, content: str) -> str:
        path = Path(directory) / name
        path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
        return str(path)

    def load_pair(self, base: str = BASE_CONFIG, dynamic: str = DYNAMIC_CONFIG):
        with tempfile.TemporaryDirectory() as tmp:
            base_path = self.write_config(tmp, "config.yaml", base)
            dynamic_path = self.write_config(tmp, "dynamic.yaml", dynamic)
            with patch.dict(os.environ, TEST_ENV, clear=False):
                return load_config_with_dynamic(
                    base_path,
                    dynamic_path=dynamic_path,
                    strict_dynamic=True,
                )

    def test_dynamic_config_adds_provider_routes_and_raw_models(self):
        config = self.load_pair()

        self.assertTrue(config.dynamic_config_applied)
        self.assertIn("deepseek2api", config.providers)
        self.assertIn("coding-auto-generated", config.routes)
        self.assertIn("deepseek2api/deepseek-v4-flash", config.routes)

        raw_route = config.routes["deepseek2api/deepseek-v4-flash"]
        self.assertEqual(raw_route.route_type, "raw_model")
        self.assertEqual(raw_route.source, "dynamic")
        self.assertEqual(raw_route.provider, "deepseek2api")
        self.assertEqual(raw_route.vendor, "deepseek")
        self.assertEqual(raw_route.upstream_model, "deepseek-v4-flash")

        entry = handlers._model_entry(
            "deepseek2api/deepseek-v4-flash",
            raw_route,
            1700000000,
            config,
        )
        self.assertEqual(entry["route_type"], "raw_model")
        self.assertEqual(entry["provider"], "deepseek2api")
        self.assertEqual(entry["vendor"], "deepseek")
        self.assertEqual(entry["model_name"], "deepseek-v4-flash")
        self.assertEqual(entry["upstream_model"], "deepseek-v4-flash")

    def test_static_routes_and_providers_win_name_conflicts(self):
        config = self.load_pair()

        self.assertEqual(
            [step.model for step in config.routes["agent-auto"].steps],
            ["static-agent-model"],
        )
        self.assertEqual(
            config.providers["openrouter"].base_url,
            "https://static-openrouter.example/v1",
        )

    def test_dynamic_config_keeps_static_server_auth_and_admin(self):
        config = self.load_pair()

        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 4999)
        self.assertEqual(config.api_keys, {"test-key"})
        self.assertEqual(config.admin_token, "test-admin")

    def test_dynamic_config_raises_max_attempts_to_longest_route(self):
        config = self.load_pair()

        self.assertEqual(config.max_attempts, 3)

    def test_empty_dynamic_routes_are_rejected(self):
        dynamic = """
        metadata:
          schema_version: 1
        providers:
          deepseek2api:
            type: openai-compatible
            base_url: https://deepseek.stpnk.tech/v1
            api_key_env: DEEPSEEK2API_API_KEY
        routes:
          agent-auto-pay: []
        """

        with self.assertRaisesRegex(ConfigError, "no non-empty routes"):
            self.load_pair(dynamic=dynamic)

    def test_unknown_provider_in_dynamic_route_is_rejected(self):
        dynamic = """
        metadata:
          schema_version: 1
        routes:
          broken:
            - provider: missing-provider
              model: model-a
        """

        with self.assertRaisesRegex(ConfigError, "unknown provider"):
            self.load_pair(dynamic=dynamic)

    def test_inline_secret_in_dynamic_yaml_is_rejected(self):
        dynamic = """
        metadata:
          schema_version: 1
        providers:
          bad:
            type: openai-compatible
            base_url: https://bad.example/v1
            api_key: sk-inline-secret
            api_key_env: BAD_API_KEY
        routes:
          bad-route:
            - provider: bad
              model: model-a
        """

        with self.assertRaisesRegex(ConfigError, "inline secret"):
            self.load_pair(dynamic=dynamic)

    def test_invalid_dynamic_yaml_keeps_base_config_in_non_strict_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_path = self.write_config(tmp, "config.yaml", BASE_CONFIG)
            dynamic_path = self.write_config(tmp, "dynamic.yaml", "not-a-mapping")
            with patch.dict(os.environ, TEST_ENV, clear=False):
                config = load_config_with_dynamic(
                    base_path,
                    dynamic_path=dynamic_path,
                    strict_dynamic=False,
                )

        self.assertFalse(config.dynamic_config_applied)
        self.assertIn("agent-auto", config.routes)
        self.assertIsNotNone(config.dynamic_config_error)

    def test_github_artifact_zip_extraction_uses_router_member(self):
        payload = b"metadata:\n  schema_version: 1\nroutes: {}\n"
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("other.txt", "ignore")
            zf.writestr("output/tinyllm-router-config.yaml", payload)

        self.assertEqual(_extract_dynamic_yaml(buf.getvalue()), payload)


class DynamicReloadTests(unittest.IsolatedAsyncioTestCase):
    def write_config(self, directory: str, name: str, content: str) -> str:
        path = Path(directory) / name
        path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
        return str(path)

    async def test_refresh_replaces_app_state_and_provider_config(self):
        updated_dynamic = DYNAMIC_CONFIG.replace(
            "deepseek-v4-pro",
            "deepseek-v4-reasoner",
        )

        with tempfile.TemporaryDirectory() as tmp:
            base_path = self.write_config(tmp, "config.yaml", BASE_CONFIG)
            dynamic_path = self.write_config(tmp, "dynamic.yaml", DYNAMIC_CONFIG)
            with patch.dict(os.environ, TEST_ENV, clear=False):
                config = load_config_with_dynamic(
                    base_path,
                    dynamic_path=dynamic_path,
                    strict_dynamic=True,
                )

                state = SimpleNamespace(config=config)
                provider = SimpleNamespace(config=config)
                app = {
                    "config": config,
                    "state": state,
                    "provider": provider,
                }
                previous = tinyllm_app._dynamic_config_signature(dynamic_path)
                Path(dynamic_path).write_text(
                    textwrap.dedent(updated_dynamic).strip() + "\n",
                    encoding="utf-8",
                )

                new_signature = await tinyllm_app._refresh_dynamic_config_if_needed(
                    app,
                    previous,
                )

        self.assertNotEqual(new_signature, previous)
        self.assertIs(app["config"], state.config)
        self.assertIs(app["config"], provider.config)
        self.assertIn("deepseek2api/deepseek-v4-reasoner", app["config"].routes)


if __name__ == "__main__":
    unittest.main()
