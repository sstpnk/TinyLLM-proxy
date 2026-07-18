"""Entry point — ``python -m tinyllm``.

Parses CLI args, loads config, starts the aiohttp server.
"""

from __future__ import annotations

import argparse
import logging
import sys

from aiohttp import web

from . import __version__
from .app import create_app
from .config import ConfigError, load_config

_LOG = logging.getLogger("tinyllm")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tinyllm",
        description="Lightweight OpenAI-compatible proxy with multi-provider fallback.",
    )
    p.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    p.add_argument("--host", help="Override server host from config")
    p.add_argument("--port", type=int, help="Override server port from config")
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug-level logging"
    )
    p.add_argument(
        "--version", action="version", version=f"tinyllm {__version__}"
    )
    return p.parse_args(argv)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = (
        "%(asctime)s pid=%(process)d %(name)s %(levelname)s %(message)s"
    )
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    # Load config
    try:
        config = load_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        _LOG.error("Config error: %s", exc)
        sys.exit(1)

    # CLI overrides
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    # Build & start
    app = create_app(config)
    _LOG.info(
        "TinyLLM v%s starting on %s:%s",
        __version__,
        config.host,
        config.port,
    )

    try:
        web.run_app(
            app,
            host=config.host,
            port=config.port,
            print=lambda *a: _LOG.info(a[0] if a else ""),
        )
    except KeyboardInterrupt:
        _LOG.info("Shutting down")


if __name__ == "__main__":
    main()
