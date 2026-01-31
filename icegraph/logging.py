# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Dict, Any, Optional
import json
import logging
import logging.config
import sys
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """JSON formatter for logs."""
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "processName": record.processName,
            "thread": record.thread,
            "threadName": record.threadName
        }

        # grab any exception info
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class DynamicStderrHandler(logging.StreamHandler):
    """
    A StreamHandler that always writes to the current sys.stderr.

    Allows Rich Live's redirect_stderr=True to capture logging output
    even if logging was configured before Live started.
    """
    def __init__(self, level=logging.NOTSET):
        # StreamHandler accepts stream=None, but will still use self.stream in emit;
        # we override emit to set it each time.
        super().__init__(stream=None)
        self.setLevel(level)

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr  # follow Live redirection
        super().emit(record)


def configure_logging(
    *,
    level:          str             = "INFO",
    log_file:       Optional[str]   = None,
    json_logs:      bool            = False,
    max_bytes:      int             = 10_000_000,
    backup_count:   int             = 5,
    console:        bool            = True
) -> logging.Logger:
    """
    Configure logging for the ``icegraph`` package namespace.

    This function configures handlers and formatters for the ``icegraph``
    logger and all ``icegraph.*`` sub-loggers. It is intended to be called
    explicitly by an application or entry point and does not modify the
    root logger.

    Args:
        level (str): Minimum logging level for the ``icegraph`` logger.
            Must be one of ``{"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}``
            (case-insensitive). Defaults to ``"INFO"``.
        log_file (Optional[str]): Path to a log file. If provided, a rotating
            file handler is added and parent directories are created
            automatically. Defaults to ``None``.
        json_logs (bool): If ``True``, emit logs using a JSON formatter.
            If ``False``, use a human-readable text format. Defaults to
            ``False``.
        max_bytes (int): Maximum size in bytes of the log file before rotation
            occurs. Only applies when ``log_file`` is provided. Defaults to
            ``10_000_000``.
        backup_count (int): Number of rotated log files to keep. Only applies
            when ``log_file`` is provided. Defaults to ``5``.
        console (bool): If ``True``, emits logs to the console. Defaults to ``True``.
    """

    # normalize inputs
    level = level.upper()

    # verify the level is valid
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid_levels:
        raise ValueError(
            f"Invalid log level '{level}'. Must be one of {list(valid_levels)}."
        )

    pkg_logger_name = "icegraph"

    # Always require concurrent-log-handler so file logging is MP-safe.
    if log_file is not None:
        try:
            import concurrent_log_handler  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "File logging requires the 'concurrent-log-handler' dependency for multiprocess safety. "
                "Install it with: pip install concurrent-log-handler==0.9.28"
            ) from e

    # build formatters
    standard_fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | "
        "pid=%(process)d | %(processName)s | %(threadName)s | %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Handlers
    handlers: Dict[str, Dict[str, Any]] = {}

    if console:
        handlers["console"] = {
            "class": "icegraph.logging.DynamicStderrHandler",
            "level": level,
            "formatter": "json" if json_logs else "standard"
        }

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "()": "concurrent_log_handler.ConcurrentRotatingFileHandler",
            "level": level,
            "formatter": "json" if json_logs else "standard",
            "filename": log_file,
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf-8"
        }

    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,  # keep third-party loggers alive
        "formatters": {
            "standard": {"format": standard_fmt, "datefmt": datefmt},
            "json": {"()": JsonFormatter}
        },
        "handlers": handlers,
        "loggers": {
            # everything under icegraph.* propagates to this logger.
            pkg_logger_name: {
                "level": level,
                "handlers": list(handlers.keys()),
                "propagate": False,  # avoid duplication via root handlers
            }
        }
    }

    # clear any existing loggers (doesnt touch third party or root)
    pkg_logger = logging.getLogger(pkg_logger_name)
    pkg_logger.handlers.clear()

    logging.config.dictConfig(config)
    return logging.getLogger(pkg_logger_name)
