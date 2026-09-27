"""Configuration loader for TinyLLM.

Loads YAML config and resolves provider API keys from environment variables.
Optionally merges a generated dynamic routing layer over the local base config.
"""

from __future__ import annotations

import copy
import io
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

_DYNAMIC_CONFIG_ENV = "TINYLLM_DYNAMIC_CONFIG_PATH"
_DYNAMIC_CONFIG_URL_ENV = "TINYLLM_DYNAMIC_CONFIG_URL"
_DYNAMIC_CONFIG_TOKEN_ENV = "TINYLLM_DYNAMIC_CONFIG_BEARER_TOKEN_ENV"
_DYNAMIC_CONFIG_ZIP_MEMBER_ENV = "TINYLLM_DYNAMIC_CONFIG_ZIP_MEMBER"
_DEFAULT_ZIP_MEMBER = "output/tinyllm-router-config.yaml"
_DYNAMIC_ALLOWED_SECTIONS = {
    "metadata",
    "routing",
    "timeouts",
    "providers",
    "routes",
}
_SECRET_VALUE_PREFIXES = (
    "sk-",
    "gsk_",
    "hf_",
    "csk-",
    "r8_",
    "github_pat_",
)


class ConfigError(Exception):
    """Configuration loading/validation error."""


class ProviderConfig:
    """Configuration for a single upstream provider."""

    def __init__(self, name: str, data: dict[str, Any]) -> None:
        self.name = name
        if not isinstance(data, dict):
            raise ConfigError(f"Provider '{self.name}': config must be a mapping")
        for field in ("base_url", "api_key_env"):
            if field not in data:
                raise ConfigError(
                    f"Provider '{self.name}': missing required field {field}"
                )
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

    def __init__(
        self,
        name: str,
        steps_data: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.steps = [RouteStep(s) for s in steps_data]
        metadata = metadata or {}
        self.source: str = metadata.get("source", "static")
        self.route_type: str = metadata.get("route_type", "route")
        self.provider: str | None = metadata.get("provider")
        self.vendor: str | None = metadata.get("vendor")
        self.upstream_model: str | None = metadata.get("upstream_model")
        self.model_name: str | None = metadata.get("model_name")


class TimeoutConfig:
    """Timeout settings."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.connect_seconds: int = int(data.get("connect_seconds", 10))
        self.response_seconds: int = int(data.get("response_seconds", 180))
        self.stream_idle_seconds: int = int(data.get("stream_idle_seconds", 300))


class AppConfig:
    """Root application configuration parsed from YAML + env."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.base_config_path: str | None = data.get("_base_config_path")
        self.dynamic_config_path: str | None = data.get("_dynamic_config_path")
        self.dynamic_config_url: str | None = data.get("_dynamic_config_url")
        self.dynamic_config_applied: bool = bool(data.get("_dynamic_config_applied"))
        self.dynamic_config_error: str | None = data.get("_dynamic_config_error")

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
        if self.max_attempts < 1:
            raise ConfigError("routing.max_attempts must be >= 1")
        self.min_requests_for_trust: int = int(routing.get("min_requests_for_trust", 20))
        self.min_success_rate: float = float(routing.get("min_success_rate", 0.5))
        self.max_empty_rate: float = float(routing.get("max_empty_rate", 0.3))
        self.min_score: float = float(routing.get("min_score", 0.0))

        # --- admin ---
        admin = data.get("admin", {})
        self.admin_token_env: str = admin.get("token_env", "TINYLLM_ADMIN_TOKEN")
        self.admin_token: str | None = os.environ.get(self.admin_token_env) or None

        # --- timeouts ---
        self.timeouts = TimeoutConfig(data.get("timeouts", {}))

        # --- providers ---
        self.providers: dict[str, ProviderConfig] = {}
        for name, pdata in data.get("providers", {}).items():
            self.providers[name] = ProviderConfig(name, pdata)

        # --- routes ---
        self.routes: dict[str, Route] = {}
        route_metadata = data.get("_route_metadata", {})
        for name, steps in data.get("routes", {}).items():
            self.routes[name] = Route(name, steps, route_metadata.get(name))

        if not self.routes:
            raise ConfigError("No routes defined in config")
        if not self.providers:
            raise ConfigError("No providers defined in config")
        self._validate_routes()

    # ------------------------------------------------------------------

    def get_route(self, name: str) -> Route | None:
        return self.routes.get(name)

    def get_provider(self, name: str) -> ProviderConfig | None:
        return self.providers.get(name)

    @property
    def route_names(self) -> list[str]:
        return list(self.routes.keys())

    def _validate_routes(self) -> None:
        for route_name, route in self.routes.items():
            if not route.steps:
                raise ConfigError(f"Route '{route_name}': no steps defined")
            for idx, step in enumerate(route.steps, start=1):
                if step.provider not in self.providers:
                    raise ConfigError(
                        f"Route '{route_name}' step {idx}: "
                        f"unknown provider '{step.provider}'"
                    )


# ------------------------------------------------------------------


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load and validate configuration from a YAML file.

    When ``TINYLLM_DYNAMIC_CONFIG_PATH`` is set, the referenced generated YAML
    is validated and merged over ``path``. Invalid dynamic config is ignored by
    default so the local base config can keep serving traffic.
    """
    return load_config_with_dynamic(path)


def load_config_with_dynamic(
    path: str = "config.yaml",
    *,
    dynamic_path: str | None = None,
    strict_dynamic: bool = False,
) -> AppConfig:
    """Load base config and optionally merge a generated dynamic config."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    data = _load_yaml_mapping(path, "Config file")
    data["_base_config_path"] = str(path)
    data["_dynamic_config_url"] = os.environ.get(_DYNAMIC_CONFIG_URL_ENV) or None

    dynamic_path = dynamic_path or os.environ.get(_DYNAMIC_CONFIG_ENV) or None
    if not dynamic_path:
        return AppConfig(data)

    data["_dynamic_config_path"] = dynamic_path
    dynamic_file = Path(dynamic_path)
    if not dynamic_file.exists():
        data["_dynamic_config_error"] = f"Dynamic config file not found: {dynamic_file}"
        if strict_dynamic:
            raise ConfigError(data["_dynamic_config_error"])
        return AppConfig(data)

    try:
        dynamic_data = _load_yaml_mapping(dynamic_file, "Dynamic config file")
        merged = merge_dynamic_config(data, dynamic_data)
        merged["_base_config_path"] = str(path)
        merged["_dynamic_config_path"] = str(dynamic_file)
        merged["_dynamic_config_url"] = data["_dynamic_config_url"]
        merged["_dynamic_config_applied"] = True
        return AppConfig(merged)
    except ConfigError as exc:
        data["_dynamic_config_error"] = str(exc)
        if strict_dynamic:
            raise
        return AppConfig(data)


def merge_dynamic_config(
    base_data: dict[str, Any],
    dynamic_data: dict[str, Any],
) -> dict[str, Any]:
    """Return base config merged with a validated generated routing layer."""
    _validate_dynamic_config(base_data, dynamic_data)
    merged = copy.deepcopy(base_data)
    route_metadata: dict[str, dict[str, Any]] = copy.deepcopy(
        merged.get("_route_metadata", {})
    )

    for section in ("routing", "timeouts"):
        if section in dynamic_data:
            current = dict(merged.get(section, {}))
            current.update(dynamic_data[section] or {})
            merged[section] = current

    merged_providers = dict(merged.get("providers", {}))
    for name, provider_data in (dynamic_data.get("providers") or {}).items():
        if name not in merged_providers:
            merged_providers[name] = provider_data
    merged["providers"] = merged_providers

    merged_routes = dict(merged.get("routes", {}))
    dynamic_routes = dynamic_data.get("routes") or {}
    for route_name, steps in dynamic_routes.items():
        if not steps:
            continue
        if route_name not in merged_routes:
            merged_routes[route_name] = steps
            route_metadata[route_name] = {
                "source": "dynamic",
                "route_type": "route",
            }
        for step in steps:
            raw_name = _raw_route_name(step["provider"], step["model"])
            if raw_name in merged_routes:
                continue
            merged_routes[raw_name] = [
                {"provider": step["provider"], "model": step["model"]}
            ]
            vendor, model_name = split_model_identity(step["provider"], step["model"])
            route_metadata[raw_name] = {
                "source": "dynamic",
                "route_type": "raw_model",
                "provider": step["provider"],
                "vendor": vendor,
                "upstream_model": step["model"],
                "model_name": model_name,
            }

    merged["routes"] = merged_routes
    merged["_route_metadata"] = route_metadata
    _raise_max_attempts_to_longest_route(merged)
    return merged


def split_model_identity(provider: str, upstream_model: str) -> tuple[str, str]:
    """Return ``(vendor, model_name)`` for a provider/model identifier."""
    if "/" in upstream_model:
        vendor, model_name = upstream_model.split("/", 1)
        return vendor, model_name
    if "-" in upstream_model:
        vendor = upstream_model.split("-", 1)[0]
        return vendor, upstream_model
    return provider, upstream_model


def download_dynamic_config(target_path: str) -> bool:
    """Download a configured dynamic artifact and atomically replace target.

    Returns True when the target file changed. The URL is read from
    ``TINYLLM_DYNAMIC_CONFIG_URL``. Zip artifacts are supported; by default the
    member ``output/tinyllm-router-config.yaml`` is extracted.
    """
    url = os.environ.get(_DYNAMIC_CONFIG_URL_ENV)
    if not url:
        return False
    target = Path(target_path)
    body = _download_url(url)
    body = _extract_dynamic_yaml(body)
    previous = target.read_bytes() if target.exists() else None
    if previous == body:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_bytes(body)
    os.replace(tmp, target)
    return True


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{label} is empty or not a valid YAML mapping")
    return data


def _validate_dynamic_config(
    base_data: dict[str, Any],
    dynamic_data: dict[str, Any],
) -> None:
    if not isinstance(dynamic_data, dict):
        raise ConfigError("Dynamic config must be a YAML mapping")
    unknown_sections = set(dynamic_data) - _DYNAMIC_ALLOWED_SECTIONS
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ConfigError(f"Dynamic config contains unsupported section(s): {names}")
    _reject_inline_secrets(dynamic_data)

    metadata = dynamic_data.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ConfigError("Dynamic config metadata must be a mapping")
    if metadata.get("schema_version") != 1:
        raise ConfigError("Dynamic config metadata.schema_version must be 1")

    for section in ("routing", "timeouts"):
        if section in dynamic_data and not isinstance(dynamic_data[section], dict):
            raise ConfigError(f"Dynamic config {section} must be a mapping")

    providers = dynamic_data.get("providers") or {}
    if not isinstance(providers, dict):
        raise ConfigError("Dynamic config providers must be a mapping")
    for name, pdata in providers.items():
        if not isinstance(pdata, dict):
            raise ConfigError(f"Dynamic provider '{name}': config must be a mapping")
        if pdata.get("type") != "openai-compatible":
            raise ConfigError(
                f"Dynamic provider '{name}': type must be openai-compatible"
            )
        for field in ("base_url", "api_key_env"):
            if not pdata.get(field):
                raise ConfigError(f"Dynamic provider '{name}': missing {field}")

    routes = dynamic_data.get("routes")
    if not isinstance(routes, dict):
        raise ConfigError("Dynamic config routes must be a mapping")
    non_empty_routes = {name: steps for name, steps in routes.items() if steps}
    if not non_empty_routes:
        raise ConfigError("Dynamic config has no non-empty routes")

    known_providers = set((base_data.get("providers") or {})) | set(providers)
    for route_name, steps in non_empty_routes.items():
        if not isinstance(steps, list):
            raise ConfigError(f"Dynamic route '{route_name}': steps must be a list")
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                raise ConfigError(
                    f"Dynamic route '{route_name}' step {idx}: must be a mapping"
                )
            provider = step.get("provider")
            model = step.get("model")
            if not provider or not model:
                raise ConfigError(
                    f"Dynamic route '{route_name}' step {idx}: provider and model required"
                )
            if provider not in known_providers:
                raise ConfigError(
                    f"Dynamic route '{route_name}' step {idx}: unknown provider '{provider}'"
                )


def _reject_inline_secrets(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() == "api_key":
                raise ConfigError(f"Dynamic config contains inline secret key at {path}")
            _reject_inline_secrets(child, f"{path}.{key_text}")
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            _reject_inline_secrets(child, f"{path}[{idx}]")
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower().startswith("bearer "):
            raise ConfigError(f"Dynamic config contains bearer token at {path}")
        if any(stripped.startswith(prefix) for prefix in _SECRET_VALUE_PREFIXES):
            raise ConfigError(f"Dynamic config contains inline secret at {path}")


def _raise_max_attempts_to_longest_route(data: dict[str, Any]) -> None:
    routes = data.get("routes") or {}
    longest = max((len(steps) for steps in routes.values() if steps), default=1)
    routing = dict(data.get("routing", {}))
    current = int(routing.get("max_attempts", 3))
    if current < longest:
        routing["max_attempts"] = longest
    data["routing"] = routing


def _raw_route_name(provider: str, upstream_model: str) -> str:
    return f"{provider}/{upstream_model}"


def _download_url(url: str) -> bytes:
    headers = {"User-Agent": "TinyLLM/0.1"}
    token_env = os.environ.get(_DYNAMIC_CONFIG_TOKEN_ENV)
    if token_env:
        token = os.environ.get(token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _extract_dynamic_yaml(body: bytes) -> bytes:
    if not body.startswith(b"PK\x03\x04"):
        return body

    preferred = os.environ.get(_DYNAMIC_CONFIG_ZIP_MEMBER_ENV) or _DEFAULT_ZIP_MEMBER
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        names = zf.namelist()
        member = preferred if preferred in names else None
        if member is None:
            for name in names:
                if name.endswith("tinyllm-router-config.yaml"):
                    member = name
                    break
        if member is None:
            raise ConfigError("Dynamic artifact zip has no tinyllm-router-config.yaml")
        return zf.read(member)
