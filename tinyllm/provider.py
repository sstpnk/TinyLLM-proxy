"""Async HTTP client for upstream AI providers.

Sends requests to OpenAI-compatible APIs, classifies errors, and
determines whether each error should trigger a fallback attempt.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp

from .config import AppConfig

logger = logging.getLogger("tinyllm.provider")

# -- Error classification --------------------------------------------------

_FALLBACK_TRIGGERS = {
    "timeout",
    "connection_error",
    "rate_limited",
    "quota_exceeded",
    "auth_error",
    "not_found",
    "server_error",
}

_QUOTA_PATTERNS = (
    "quota",
    "rate_limit",
    "insufficient_quota",
    "rate limit",
    "insufficient quota",
    "exceeded",
    "out of credits",
    "billing",
)


class ProviderError(Exception):
    """A provider returned an error or the request failed."""

    def __init__(
        self,
        error_type: str,
        message: str = "",
        status_code: int = 0,
        body: str = "",
    ) -> None:
        self.error_type = error_type
        self.message = message
        self.status_code = status_code
        self.body = body
        super().__init__(f"{error_type}: {message[:120]}")

    # ------------------------------------------------------------------

    def should_fallback(self) -> bool:
        """Return True if this error warrants trying the next provider."""
        if self.error_type in _FALLBACK_TRIGGERS:
            return True
        # When status is 400, check the body for quota-related keywords
        if self.status_code == 400 and self.body:
            lower = self.body.lower()
            if any(p in lower for p in _QUOTA_PATTERNS):
                return True
        return False

    # ------------------------------------------------------------------

    @classmethod
    def from_response(cls, status: int, body: str) -> ProviderError:
        """Build a ProviderError from an HTTP response."""
        body_lower = body.lower()
        # Determine error type
        if status in (401, 403):
            error_type = "auth_error"
        elif status == 404:
            error_type = "not_found"
        elif status == 408:
            error_type = "timeout"
        elif status == 429:
            error_type = "rate_limited"
        elif 500 <= status < 600:
            error_type = "server_error"
        elif any(p in body_lower for p in _QUOTA_PATTERNS):
            error_type = "quota_exceeded"
        else:
            error_type = "request_error"

        # Extract a human-readable message from the response body
        message = body[:200]
        try:
            data = json.loads(body)
            err = data.get("error", {}) if isinstance(data, dict) else {}
            if isinstance(err, dict):
                message = err.get("message", err.get("code", body[:200]))
            elif isinstance(err, str):
                message = err
        except (json.JSONDecodeError, TypeError):
            pass

        return cls(error_type, message, status, body)

    @classmethod
    def from_exception(cls, exc: Exception) -> ProviderError:
        """Build a ProviderError from a network / timeout exception."""
        if isinstance(exc, asyncio.TimeoutError):
            return cls("timeout", "Request timed out")
        if isinstance(exc, aiohttp.ClientConnectorError):
            return cls("connection_error", f"Connection failed: {exc}")
        if isinstance(exc, aiohttp.ServerDisconnectedError):
            return cls("connection_error", "Server disconnected")
        return cls("connection_error", str(exc))


# -- Client -----------------------------------------------------------------


class ProviderClient:
    """Async HTTP client for upstream OpenAI-compatible APIs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"}
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # -- request helpers ----------------------------------------------------

    def _build_request(
        self,
        step,
        request_body: dict[str, Any],
        stream: bool,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build (url, headers, body) for a provider request."""
        provider_cfg = self.config.get_provider(step.provider)
        if provider_cfg is None:
            raise ValueError(f"Unknown provider: {step.provider}")

        url = f"{provider_cfg.base_url}/chat/completions"

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider_cfg.api_key}",
        }
        headers.update(provider_cfg.headers)

        # Clone and patch the request body
        body = dict(request_body)
        body["model"] = step.model
        if stream:
            body["stream"] = True
        elif "stream" in body:
            del body["stream"]  # don't send stream=false

        return url, headers, body

    # -- non-streaming ------------------------------------------------------

    async def send_non_streaming(
        self,
        step,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a non-streaming request.  Returns the parsed JSON response."""
        url, headers, body = self._build_request(step, request_body, stream=False)
        session = await self._get_session()
        timeout = aiohttp.ClientTimeout(
            connect=self.config.timeouts.connect_seconds,
            total=self.config.timeouts.response_seconds,
        )

        try:
            async with session.post(
                url, headers=headers, json=body, timeout=timeout
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                error_body = await resp.text()
                raise ProviderError.from_response(resp.status, error_body)

        except asyncio.TimeoutError:
            raise ProviderError("timeout", "Request timed out")
        except aiohttp.ClientError as exc:
            raise ProviderError.from_exception(exc)

    # -- streaming ----------------------------------------------------------

    async def send_streaming(
        self,
        step,
        request_body: dict[str, Any],
    ) -> aiohttp.ClientResponse:
        """Send a streaming request.

        Returns the *aiohttp.ClientResponse* on success (status 200).
        The **caller** must iterate over ``resp.content`` and call
        ``resp.release()`` when done.
        """
        url, headers, body = self._build_request(step, request_body, stream=True)
        session = await self._get_session()
        timeout = aiohttp.ClientTimeout(
            connect=self.config.timeouts.connect_seconds,
            total=None,  # no total timeout — per-chunk idle timeout is used instead
        )

        try:
            resp = await session.post(
                url, headers=headers, json=body, timeout=timeout
            )
            if resp.status == 200:
                return resp

            error_body = await resp.text()
            await resp.release()
            raise ProviderError.from_response(resp.status, error_body)

        except asyncio.TimeoutError:
            raise ProviderError("timeout", "Request timed out")
        except aiohttp.ClientError as exc:
            raise ProviderError.from_exception(exc)
