"""HTTP request handlers for TinyLLM.

Implements:
  - POST /v1/chat/completions   (non-streaming + streaming with fallback)
  - GET  /v1/models
  - GET  /health/liveliness
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from aiohttp import web

from .provider import ProviderClient, ProviderError
from .state import AppState

logger = logging.getLogger("tinyllm.handlers")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_request_id_counter = 0


def _new_id() -> str:
    """Return a short unique request id."""
    global _request_id_counter
    _request_id_counter += 1
    return f"req-{uuid.uuid4().hex[:8]}-{_request_id_counter}"


def _openai_error(
    status: int,
    message: str,
    error_type: str = "server_error",
    code: str | None = None,
) -> web.Response:
    """Build an OpenAI-compatible JSON error response."""
    error: dict[str, Any] = {"message": message, "type": error_type}
    if code:
        error["code"] = code
    return web.json_response({"error": error}, status=status)


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


async def handle_chat_completions(request: web.Request) -> web.Response:
    """Route a completion request through the fallback chain."""
    state: AppState = request.app["state"]
    provider: ProviderClient = request.app["provider"]

    # --- parse body ---
    try:
        body: dict[str, Any] = await request.json()
    except json.JSONDecodeError:
        return _openai_error(400, "Invalid JSON", "invalid_request_error")

    model = body.get("model", "")
    if not model:
        return _openai_error(400, "Missing field: model", "invalid_request_error")

    route = state.config.get_route(model)
    if route is None:
        return _openai_error(
            404,
            f"Model '{model}' not found",
            "invalid_request_error",
            "model_not_found",
        )

    stream = body.get("stream", False)
    rid = _new_id()

    if stream:
        return await _handle_streaming(request, body, route, rid, state, provider)
    return await _handle_non_streaming(body, route, rid, state, provider)


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


async def _handle_non_streaming(
    body: dict[str, Any],
    route,
    rid: str,
    state: AppState,
    provider: ProviderClient,
) -> web.Response:
    """Non-streaming: try providers in order, return first success."""
    client_model = body.get("model", "")

    for step in route.steps:
        if state.is_cooldown_active(step.provider, step.model):
            continue

        start = time.monotonic()
        try:
            data = await provider.send_non_streaming(step, body)
        except ProviderError as exc:
            latency = (time.monotonic() - start) * 1000
            logger.info(
                "request=%s route=%s provider=%s model=%s "
                "attempt=%d status=%s latency=%.0fms fallback=%s",
                rid,
                route.name,
                step.provider,
                step.model,
                _attempt_index(route, step),
                exc.error_type,
                latency,
                _next_provider(route, step),
            )
            state.metrics.total_fallbacks += 1
            state.mark_error(step.provider, step.model, exc.error_type)

            if not exc.should_fallback():
                return _openai_error(
                    exc.status_code or 400,
                    exc.message or "Provider error",
                    "provider_error",
                )
            continue  # try next

        # --- success ---
        latency = (time.monotonic() - start) * 1000
        state.metrics.total_requests += 1
        state.metrics.successful_requests += 1
        state.metrics.total_latency_ms += latency
        state.mark_success(step.provider, step.model)

        logger.info(
            "request=%s route=%s provider=%s model=%s "
            "status=200 latency=%.0fms",
            rid,
            route.name,
            step.provider,
            step.model,
            latency,
        )

        # Override model so the client sees its own model name
        if isinstance(data, dict) and "model" in data:
            data["model"] = client_model

        return web.json_response(
            data,
            headers={
                "X-Request-Id": rid,
                "X-Provider": step.provider,
                "X-Model": step.model,
            },
        )

    state.metrics.total_requests += 1
    return _openai_error(503, "All providers failed", "server_error", "all_failed")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def _handle_streaming(
    request: web.Request,
    body: dict[str, Any],
    route,
    rid: str,
    state: AppState,
    provider: ProviderClient,
) -> web.Response:
    """Streaming: try providers in order, forward SSE chunks on first success."""
    for step in route.steps:
        if state.is_cooldown_active(step.provider, step.model):
            continue

        try:
            upstream = await provider.send_streaming(step, body)
        except ProviderError as exc:
            logger.info(
                "request=%s route=%s provider=%s model=%s "
                "attempt=%d status=%s fallback=%s",
                rid,
                route.name,
                step.provider,
                step.model,
                _attempt_index(route, step),
                exc.error_type,
                _next_provider(route, step),
            )
            state.metrics.total_fallbacks += 1
            state.mark_error(step.provider, step.model, exc.error_type)

            if not exc.should_fallback():
                return _openai_error(
                    exc.status_code or 400,
                    exc.message or "Provider error",
                    "provider_error",
                )
            continue

        # -- upstream connected, start streaming to client -- #
        state.metrics.total_requests += 1

        start = time.monotonic()
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-Id": rid,
                "X-Provider": step.provider,
                "X-Model": step.model,
            },
        )
        await resp.prepare(request)

        success = False
        try:
            await _forward_stream(
                upstream,
                resp,
                rid,
                state.config.timeouts.stream_idle_seconds,
                client_model=body.get("model", ""),
            )
            success = True
        except (ConnectionResetError, ConnectionAbortedError):
            logger.debug("request=%s client disconnected", rid)
        finally:
            upstream.release()

        if success:
            latency = (time.monotonic() - start) * 1000
            state.metrics.total_latency_ms += latency
            state.metrics.successful_requests += 1
            state.mark_success(step.provider, step.model)
            logger.info(
                "request=%s route=%s provider=%s model=%s "
                "status=200 latency=%.0fms stream=1",
                rid,
                route.name,
                step.provider,
                step.model,
                latency,
            )

        return resp

    state.metrics.total_requests += 1
    return _openai_error(503, "All providers failed", "server_error", "all_failed")


async def _forward_stream(
    upstream,  # aiohttp.ClientResponse
    downstream: web.StreamResponse,
    rid: str,
    idle_timeout: int,
    *,
    client_model: str = "",
) -> None:
    """Forward SSE chunks from upstream to the client with idle timeout."""
    while True:
        try:
            chunk = await asyncio.wait_for(
                upstream.content.readany(),
                timeout=idle_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("request=%s stream idle timeout", rid)
            break

        if not chunk:  # EOF
            break

        # Split chunk into SSE events (separated by \n\n) and process each
        if client_model:
            chunk = _process_sse_buffer(chunk, client_model)
            # Log multi-event buffers for diagnostics
            if chunk.count(b"\n\n") > 1:
                first = chunk.split(b"\n\n")[0]
                logger.info(
                    "SSE_BUF: multi-event buf events=%d first=%.80s",
                    chunk.count(b"\n\n"),
                    first.decode("utf-8", errors="replace")[:80],
                )

        await downstream.write(chunk)
        await downstream.drain()

    # Signal stream end
    try:
        await downstream.write(b"data: [DONE]\n\n")
        await downstream.drain()
    except (ConnectionResetError, ConnectionAbortedError):
        logger.debug("request=%s client disconnected before final [DONE]", rid)


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


async def handle_list_models(request: web.Request) -> web.Response:
    """Return configured routes as available models."""
    state: AppState = request.app["state"]

    data = [
        {
            "id": name,
            "object": "model",
            "created": int(state._start_time),
            "owned_by": "tinyllm",
        }
        for name in state.config.route_names
    ]

    return web.json_response({"object": "list", "data": data})


# ---------------------------------------------------------------------------
# GET /health/liveliness
# ---------------------------------------------------------------------------


async def handle_health(request: web.Request) -> web.Response:
    """Liveness probe — always returns 200 when the service is alive."""
    return web.json_response({"status": "ok"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _process_sse_buffer(buffer: bytes, client_model: str) -> bytes:
    """Process a raw SSE byte buffer, overriding model in every ``data:`` line.

    ``readany()`` may return multiple SSE events concatenated in one buffer.
    We split by ``\\n\\n`` to isolate individual events, process each, and
    reassemble.  Events containing ``[DONE]`` are dropped.
    """
    events = buffer.split(b"\n\n")
    if len(events) > 2:
        logger.info(
            "BUF_DEBUG: %d events, total_len=%d, first=%.60s",
            len(events),
            len(buffer),
            events[0].decode("utf-8", errors="replace")[:60] if events[0] else "?",
        )
    parts: list[bytes] = []
    for raw_event in events:
        if not raw_event or b"[DONE]" in raw_event:
            continue
        event = raw_event.decode("utf-8")
        # Process only the last (data:) line of a multi-line event
        lines = event.split("\n")
        data_line = None
        for ln in lines:
            if ln.startswith("data:") or ln.lstrip().startswith("data:"):
                data_line = ln
        if data_line is not None:
            modified = _replace_model_in_sse(data_line, client_model)
            if modified is not None:
                parts.append(modified)
                continue
        parts.append(raw_event + b"\n\n")
    return b"".join(parts)


def _replace_model_in_sse(event: str, model: str) -> bytes | None:
    """Replace ``model`` field in a single SSE ``data:`` line with *model*.

    Strips leading whitespace/newlines before ``data:`` so that
    fragmented TCP reads still get their model overridden.

    Returns the modified event bytes (with ``\\n\\n`` terminator), or ``None``
    if no change was needed.
    """
    stripped = event.strip("\r\n ")
    if not stripped.startswith("data:") or "[DONE]" in stripped:
        return None
    try:
        # Handle "data: {...}" or "data:{...}"
        json_part = stripped[5:]
        if json_part.startswith(" "):
            json_part = json_part[1:]
        payload = json.loads(json_part, strict=False)
        if "model" in payload:
            payload["model"] = model
            result = ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode(
                "utf-8"
            )
            return result
        # Log when model field is missing
        logger.info("RM_DEBUG: no model in data | has_keys=%s", list(payload.keys()))
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.info("RM_DEBUG: json err %s | event=%.120s", exc, event[:120])
    return None


def _attempt_index(route, step) -> int:
    """Return 1-based attempt index for *step* in *route*."""
    try:
        return route.steps.index(step) + 1
    except ValueError:
        return len(route.steps)


def _next_provider(route, step) -> str:
    """Return the name of the next provider in the route, or 'none'."""
    try:
        idx = route.steps.index(step)
        if idx + 1 < len(route.steps):
            return route.steps[idx + 1].provider
    except ValueError:
        pass
    return "none"
