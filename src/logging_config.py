"""Application logging configuration.

The mu-python-template logger already owns a file handler and a stdout
handler.  Letting it propagate to a separately configured root handler emits
every template record twice and, because the handlers use different streams,
can interleave large multiline records in container logs.
"""

import logging
import os
import sys
from typing import TextIO


DEFAULT_LOG_FORMAT = "%(levelname)s:%(name)s:%(message)s"


def _log_level(level_name: str | None) -> int:
    """Resolve a configured log level, falling back safely to WARNING."""
    name = (level_name or "WARNING").strip().upper()
    level = getattr(logging, name, None)
    return level if isinstance(level, int) else logging.WARNING


def configure_logging(
    template_logger: logging.Logger,
    *,
    level_name: str | None = None,
    root_logger: logging.Logger | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure application logs without duplicating template records.

    ``root_logger`` and ``stream`` are injectable to keep this behavior easy
    to verify without mutating process-global logging state in tests.
    """
    level = _log_level(level_name or os.environ.get("LOG_LEVEL"))
    root = root_logger or logging.getLogger()
    root.setLevel(level)

    # Uvicorn or another host may already have configured the root logger.  In
    # that case, preserve the host's handlers instead of adding a second one.
    if not root.handlers:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        root.addHandler(handler)

    template_logger.setLevel(level)
    template_logger.propagate = False
