"""TinyLLM web application — server setup and auth middleware."""

from __future__ import annotations

import logging

from aiohttp import web

from .config import AppConfig
from .handlers import handle_chat_completions, handle_health, handle_list_models
from .provider import ProviderClient
from .state import AppState

logger = logging.getLogger("tinyllm")

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

    async def _cleanup(app: web.Application) -> None:
        provider: ProviderClient | None = app.get("provider")
        if provider:
            await provider.close()

    app.on_startup.append(_init_provider)
    app.on_cleanup.append(_cleanup)

    # --- routes ---
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_list_models)
    app.router.add_get("/health/liveliness", handle_health)

    logger.info(
        "App created: %d route(s), %d provider(s), %d api key(s)",
        len(config.routes),
        len(config.providers),
        len(config.api_keys),
    )
    return app


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


@web.middleware
async def _auth_middleware(
    request: web.Request, handler: web.RequestHandler
) -> web.Response:
    """Validate Bearer token on all endpoints except /health/liveliness."""
    if request.path == "/health/liveliness":
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
    if key not in config.api_keys:
        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "auth_error"}},
            status=401,
        )

    return await handler(request)
