"""Central logging configuration.

Call :func:`configure_logging` once at process startup (done in ``main.py``'s
lifespan). Everything else just uses ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a concise, readable format.

    Idempotent: safe to call more than once (e.g. under --reload) without
    stacking duplicate handlers.
    """
    root = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    # Avoid duplicate handlers if configure_logging runs twice.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)

    # Quiet down noisy third-party loggers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
