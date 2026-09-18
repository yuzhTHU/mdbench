# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""Shared logging utilities for MDBench.

Library modules:

    from src.utils.logger import logger

    logger.info("Started")
    logger.warning("Something happened")

Entry-point configuration:

    from src.utils.logger import config_logger

    config_logger(
        console_level="debug",
        exp_name="MyExp",
        save_path="logs/train.log",
    )
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from datetime import timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import MethodType


__all__ = [
    "logger",
    "config_logger",
    "LogFormatter",
    "TRACE",
    "NOTE",
]


TRACE = 15
NOTE = 25

logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(NOTE, "NOTE")

_LEVELS = {
    "debug": logging.DEBUG,
    "trace": TRACE,
    "info": logging.INFO,
    "note": NOTE,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        self._log(TRACE, message, args, **kwargs)


def _note(self, message, *args, **kwargs):
    if self.isEnabledFor(NOTE):
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        self._log(NOTE, message, args, **kwargs)


class LogFormatter(logging.Formatter):
    """Format logs with colors, timestamps, and elapsed time."""

    COLORS = {
        "DEBUG": "\033[0;37m{}\033[0m",
        "TRACE": "\033[1;48;5;240m{}\033[0m",
        "INFO": "\033[0;34m{}\033[0m",
        "NOTE": "\033[1;38;5;46m{}\033[0m",
        "WARNING": "\033[1;48;5;220m{}\033[0m",
        "ERROR": "\033[0;30;41m{}\033[0m",
        "CRITICAL": "\033[0;30;45m{}\033[0m",
    }

    def __init__(
        self,
        exp_name: str = "MDBench",
        colorful: bool = True,
        start_time: float | None = None,
        show_lineno_for: tuple[str, ...] = (
            "TRACE",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ),
    ):
        super().__init__()
        self.exp_name = exp_name
        self.colorful = colorful
        self.start_time = time.time() if start_time is None else start_time
        self.show_lineno_for = set(show_lineno_for)

    def format(self, record: logging.LogRecord) -> str:
        timestamp = time.strftime(
            "%b%d %H:%M:%S",
            time.localtime(record.created),
        )
        elapsed = timedelta(
            seconds=max(0, record.created - self.start_time)
        )

        prefix = (
            f"[{self.exp_name}|{record.module}|"
            f"{record.levelname[0]}|{timestamp}|{elapsed}]"
        )

        if record.levelname in self.show_lineno_for:
            try:
                path = os.path.relpath(record.pathname, Path.cwd())
            except ValueError:
                # relpath may fail across Windows drives.
                path = record.pathname

            prefix += f" ({path}:{record.lineno})"

        message = record.getMessage() or ""

        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        # indent = " " * (len(prefix) + 1)
        indent = " " * 4
        message = message.replace("\n", "\n" + indent)

        if not self.colorful:
            return f"{prefix} {_ANSI_ESCAPE.sub('', message)}"

        colored_prefix = self.COLORS.get(
            record.levelname,
            "{}",
        ).format(prefix)

        return f"{colored_prefix} {message}"


# Derive the logger name from this module path.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Add NOTE and TRACE only to this logger instance.
logger.trace = MethodType(_trace, logger)
logger.note = MethodType(_note, logger)


def _build_handlers(
    console_level,
    file_level,
    exp_name,
    save_path,
    colorful,
    file_max_size_mb,
    file_backup_count,
    show_lineno_for_all_levels,
):
    def parse_level(value):
        if isinstance(value, int):
            return value

        try:
            return _LEVELS[value.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown log level: {value!r}") from exc

    start_time = time.time()

    lineno_levels = [
        "TRACE",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ]
    if show_lineno_for_all_levels:
        lineno_levels += [
            "DEBUG",
            "INFO",
            "NOTE",
        ]

    handlers = []

    if console_level is not None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(parse_level(console_level))
        console_handler.setFormatter(
            LogFormatter(
                exp_name=exp_name,
                colorful=colorful,
                start_time=start_time,
                show_lineno_for=tuple(lineno_levels),
            )
        )
        handlers.append(console_handler)

    if save_path is not None:
        path = Path(save_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            path,
            maxBytes=int(file_max_size_mb * 1024 * 1024),
            backupCount=file_backup_count,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setLevel(parse_level(file_level))
        file_handler.setFormatter(
            LogFormatter(
                exp_name=exp_name,
                colorful=False,
                start_time=start_time,
                show_lineno_for=tuple(lineno_levels),
            )
        )
        handlers.append(file_handler)

    return handlers


def config_logger(
    log_queue=None,
    *,
    console_level="info",
    file_level="debug",
    exp_name="MDBench",
    save_path=None,
    colorful=True,
    file_max_size_mb=50,
    file_backup_count=10,
    show_lineno_for_all_levels=False,
    multiprocess=False,
    mp_context=None,
):
    """Configure the shared logger.

    Regular entry point:

        config_logger(
            console_level="debug",
            save_path="logs/train.log",
        )

    Multiprocessing parent:

        queue, listener = config_logger(
            save_path="logs/train.log",
            multiprocess=True,
        )

    Multiprocessing worker:

        config_logger(queue)
    """

    # Reconfiguration replaces existing handlers.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Workers only forward LogRecords through the supplied queue.
    if log_queue is not None:
        from logging.handlers import QueueHandler

        logger.addHandler(QueueHandler(log_queue))
        return None

    handlers = _build_handlers(
        console_level=console_level,
        file_level=file_level,
        exp_name=exp_name,
        save_path=save_path,
        colorful=colorful,
        file_max_size_mb=file_max_size_mb,
        file_backup_count=file_backup_count,
        show_lineno_for_all_levels=show_lineno_for_all_levels,
    )

    # Regular mode writes directly without a queue or listener.
    if not multiprocess:
        for handler in handlers:
            logger.addHandler(handler)
        return None

    # The parent serializes multiprocess records through one listener.
    import multiprocessing as mp
    from logging.handlers import QueueHandler, QueueListener

    context = mp_context or mp.get_context()
    queue = context.Queue(-1)

    listener = QueueListener(
        queue,
        *handlers,
        respect_handler_level=True,
    )
    listener.start()

    # Parent records use the same queue to preserve ordering.
    logger.addHandler(QueueHandler(queue))

    return queue, listener


# Provide colored INFO output before explicit configuration.
if not logger.handlers:
    default_handler = _build_handlers(
        console_level="info",
        file_level="debug",
        exp_name="MDBench",
        save_path=None,
        colorful=True,
        file_max_size_mb=50,
        file_backup_count=10,
        show_lineno_for_all_levels=False,
    )[0]

    logger.addHandler(default_handler)
