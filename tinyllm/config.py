"""Configuration loader for TinyLLM.

Loads YAML config and resolves provider API keys from environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Configuration loading/validation error."""


class ProviderConfig:
    """Configuration for a single upstream provider."""

    def __init__(self, name: str, data: dict[str, Any]) -> None:
        self.name = name
        self.type: str = data.get("type", "openai-compatible")
        self.base_url: str = data["base_url"].rstrip("/")
        self.api_key_env: str = data["api_key_env"]
        self.headers: dict[str, str] = data.get("headers", {})

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ConfigError(
                f"Provider '{self.name}': missing env var {self.api_key_env}"
            )
        return key


class RouteStep:
    """A single step in a route — a provider + model pair."""

    def __init__(self, data: dict[str, str]) -> None:
        self.provider: str = data["provider"]
        self.model: str = data["model"]


class Route:
    """A named route with ordered fallback steps."""

    def __init__(self, name: str, steps_data: list[dict[str, str]]) -> None:
        self.name = name
        self.steps = [RouteStep(s) for s in steps_data]


class TimeoutConfig:
    """Timeout settings."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.connect_seconds: int = int(data.get("connect_seconds", 10))
        self.response_seconds: int = int(data.get("response_seconds", 180))
        self.stream_idle_seconds: int = int(data.get("stream_idle_seconds", 300))


class AppConfig:
    """Root application configuration parsed from YAML + env."""

    def __init__(self, data: dict[str, Any]) -> None:
        # --- server ---
        server = data.get("server", {})
        self.host: str = server.get("host", "127.0.0.1")
        self.port: int = int(server.get("port", 4000))

        # --- auth ---
        auth = data.get("auth", {})
        api_keys_env: str = auth.get("api_keys_env", "TINYLLM_API_KEYS")
        keys_str = os.environ.get(api_keys_env, "")
        self.api_keys: set[str] = {
            k.strip() for k in keys_str.split(",") if k.strip()
        }
        if not self.api_keys:
            raise ConfigError(f"No API keys found in env var {api_keys_env}")

        # --- routing ---
        routing = data.get("routing", {})
        self.cooldown_seconds: int = int(routing.get("cooldown_seconds", 300))
        self.max_attempts: int = int(routing.get("max_attempts", 3))

        # --- timeouts ---
        self.timeouts = TimeoutConfig(data.get("timeouts", {}))

        # --- providers ---
        self.providers: dict[str, ProviderConfig] = {}
        for name, pdata in data.get("providers", {}).items():
            self.providers[name] = ProviderConfig(name, pdata)

        # --- routes ---
        self.routes: dict[str, Route] = {}
        for name, steps in data.get("routes", {}).items():
            self.routes[name] = Route(name, steps)

        if not self.routes:
            raise ConfigError("No routes defined in config")
        if not self.providers:
            raise ConfigError("No providers defined in config")

    # ------------------------------------------------------------------

    def get_route(self, name: str) -> Route | None:
        return self.routes.get(name)

    def get_provider(self, name: str) -> ProviderConfig | None:
        return self.providers.get(name)

    @property
    def route_names(self) -> list[str]:
        return list(self.routes.keys())


# ------------------------------------------------------------------


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load and validate configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError("Config file is empty or not a valid YAML mapping")
    return AppConfig(data)
