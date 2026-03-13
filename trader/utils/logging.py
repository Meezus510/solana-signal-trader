"""
trader/utils/logging.py — Structured logging configuration.

Call configure_logging() once at process startup. All modules obtain their
logger via logging.getLogger(__name__) — no other setup is needed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(
    level: int = logging.INFO,
    log_file: str | None = "trader.log",
) -> None:
    """
    Configure the root logger with console + optional file output.

    Also sets up two dedicated structured loggers:
        signals  → signals.log   (every Telegram message + outcome)
        trades   → trades.log    (every buy / sell event)

    Format uses a fixed-width level name so log columns align cleanly
    when tailing in a terminal.
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    handlers: list[logging.Handler] = [_stream_handler(formatter)]
    if log_file:
        handlers.append(_file_handler(log_file, formatter))

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Pipe-delimited structured logs — easy to tail or import into a spreadsheet
    _add_dedicated_logger("signals", "signals.log")
    _add_dedicated_logger("trades",  "trades.log")


def _add_dedicated_logger(name: str, path: str) -> None:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False          # keep out of console / trader.log
    if not log.handlers:
        h = _file_handler(path, logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(h)


def _stream_handler(formatter: logging.Formatter) -> logging.StreamHandler:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(formatter)
    return h


def _file_handler(path: str, formatter: logging.Formatter) -> logging.FileHandler:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(formatter)
    return h
