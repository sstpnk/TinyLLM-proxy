"""TinyLLM web application — server setup and auth middleware."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from aiohttp import web

from .config import AppConfig
from .config import ConfigError, download_dynamic_config, load_config_with_dynamic
from .handlers import (
    handle_chat_completions,
    handle_health,
    handle_list_models,
    handle_readiness,
)
from .provider import ProviderClient
from .state import AppState

logger = logging.getLogger("tinyllm")

_AUTH_EXEMPT_PATHS = {
 "/health/liveliness",
 "/health/readiness",
 "/tinyllm/v1/models",
}

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(config: AppConfig) -> web.Application:
    """Build and return a fully configured aiohttp web application."""
    app = web.Application(middlewares=[_auth_middleware])

    # Plain dict storage (aiohttp convention for app-scoped data)
    app["config"] = config
    app["state"] = AppState(config)

    async def _init_provider(app: web.Application) -> None:
        app["provider"] = ProviderClient(config)
        task = _maybe_start_dynamic_config_poller(app)
        if task:
            app["dynamic_config_task"] = task

    async def _cleanup(app: web.Application) -> None:
        task: asyncio.Task | None = app.get("dynamic_config_task")
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        provider: ProviderClient | None = app.get("provider")
        if provider:
            await provider.close()

    app.on_startup.append(_init_provider)
    app.on_cleanup.append(_cleanup)

    # --- routes ---
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_list_models)
    app.router.add_get("/tinyllm/v1/models", handle_list_models)
    app.router.add_get("/health/liveliness", handle_health)
    app.router.add_get("/health/readiness", handle_readiness)

    if config.admin_token:
        from .handlers import handle_admin_upstreams
        app.router.add_get("/v1/admin/upstreams", handle_admin_upstreams)

    logger.info(
        "App created: %d route(s), %d provider(s), %d api key(s)",
        len(config.routes),
        len(config.providers),
        len(config.api_keys),
    )
    return app


# ---------------------------------------------------------------------------
# Dynamic config polling
# ---------------------------------------------------------------------------


def _maybe_start_dynamic_config_poller(app: web.Application) -> asyncio.Task | None:
    config: AppConfig = app["config"]
    if not config.dynamic_config_path:
        if config.dynamic_config_url:
            logger.warning(
                "%s is set but %s is missing; dynamic config download disabled",
                "TINYLLM_DYNAMIC_CONFIG_URL",
                "TINYLLM_DYNAMIC_CONFIG_PATH",
            )
        return None

    interval = _dynamic_poll_interval()
    logger.info(
        "dynamic_config polling enabled path=%s interval=%.1fs url=%s",
        config.dynamic_config_path,
        interval,
        bool(config.dynamic_config_url),
    )
    return asyncio.create_task(_dynamic_config_poller(app, interval))


def _dynamic_poll_interval() -> float:
    raw = os.environ.get("TINYLLM_DYNAMIC_CONFIG_POLL_SECONDS", "30")
    try:
        return max(1.0, float(raw))
    except ValueError:
        logger.warning("invalid TINYLLM_DYNAMIC_CONFIG_POLL_SECONDS=%r; using 30", raw)
        return 30.0


async def _dynamic_config_poller(app: web.Application, interval: float) -> None:
    signature = _dynamic_config_signature(app["config"].dynamic_config_path)
    while True:
        await asyncio.sleep(interval)
        signature = await _refresh_dynamic_config_if_needed(app, signature)


async def _refresh_dynamic_config_if_needed(
    app: web.Application,
    previous_signature: tuple[int, int] | None,
) -> tuple[int, int] | None:
    current: AppConfig = app["config"]
    dynamic_path = current.dynamic_config_path
    if not dynamic_path:
        return previous_signature

    downloaded = False
    if current.dynamic_config_url:
        try:
            downloaded = await asyncio.to_thread(download_dynamic_config, dynamic_path)
        except Exception as exc:  # noqa: BLE001 - keep serving old config
            logger.warning("dynamic_config download failed: %s", exc)

    signature = _dynamic_config_signature(dynamic_path)
    if signature is None:
        return previous_signature
    if not downloaded and signature == previous_signature:
        return previous_signature

    try:
        new_config = await asyncio.to_thread(
            load_config_with_dynamic,
            current.base_config_path or "config.yaml",
            dynamic_path=dynamic_path,
            strict_dynamic=True,
        )
    except (ConfigError, OSError) as exc:
        logger.warning("dynamic_config reload rejected: %s", exc)
        return previous_signature

    app["config"] = new_config
    state = app["state"]
    state.config = new_config
    provider: ProviderClient = app["provider"]
    provider.config = new_config
    logger.info(
        "dynamic_config applied routes=%d providers=%d path=%s",
        len(new_config.routes),
        len(new_config.providers),
        dynamic_path,
    )
    return signature


def _dynamic_config_signature(path: str | None) -> tuple[int, int] | None:
    if not path:
        return None
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


@web.middleware
async def _auth_middleware(
    request: web.Request, handler: web.RequestHandler
) -> web.Response:
    """Validate Bearer token on all endpoints except /health/liveliness."""
    if request.path in _AUTH_EXEMPT_PATHS:
        return await handler(request)

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return web.json_response(
            {
                "error": {
                    "message": "Missing or malformed Authorization header",
                    "type": "auth_error",
                }
            },
            status=401,
        )

    key = auth[7:]
    config: AppConfig = request.app["config"]
    if request.path.startswith("/v1/admin/") and config.admin_token:
        if key != config.admin_token:
            return web.json_response(
                {"error": {"message": "Invalid admin token", "type": "auth_error"}},
                status=401,
            )
        return await handler(request)
    if key not in config.api_keys:
        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "auth_error"}},
            status=401,
        )

    return await handler(request)
