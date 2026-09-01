"""Centralised logging configuration for TinyLLM.

Sets up a rotating file handler plus a stderr stream handler so that logs
are both persisted on disk (surviving container restarts when the file is
bind-mounted) and surfaced via ``docker logs``.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(log_file: Path, level: str = "INFO") -> None:
    """Configure root + ``tinyllm.*`` loggers.

    Replaces any handlers previously attached to the root logger so the
    function is safe to call more than once (e.g. from tests).
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.setLevel(level.upper())

    # aiohttp has its own loggers that ignore root by default; bring them under root.
    logging.getLogger("aiohttp.access").propagate = True
    logging.getLogger("aiohttp.server").propagate = True
