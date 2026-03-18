from collections.abc import Callable
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, Filter, Formatter, LogRecord, captureWarnings, getLogger
from threading import main_thread
from typing import TypeGuard

from jetpytools import fallback
from PySide6.QtCore import QMessageLogContext, QtMsgType, qInstallMessageHandler
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

console = Console(stderr=True)
main_thread_name = main_thread().name


def _is_lambda(obj: object) -> TypeGuard[Callable[[], object]]:
    return callable(obj) and getattr(obj, "__name__", None) == "<lambda>"


def _qt_message_handler(mode: QtMsgType, context: QMessageLogContext, message: str) -> None:
    level_map = {
        QtMsgType.QtDebugMsg: DEBUG,
        QtMsgType.QtInfoMsg: INFO,
        QtMsgType.QtWarningMsg: WARNING,
        QtMsgType.QtCriticalMsg: ERROR,
        QtMsgType.QtFatalMsg: CRITICAL,
        QtMsgType.QtSystemMsg: CRITICAL,
    }

    category = context.category or "default"

    if not category.startswith("qt."):
        category = f"qt.{category}"

    level = level_map[mode]

    # Demote spammy FFmpeg version info to DEBUG
    if category == "qt.multimedia.ffmpeg" and level == INFO and "FFmpeg version" in message:
        level = DEBUG

    getLogger(category).log(level, message, stacklevel=2)


class EffectiveLevelFilter(Filter):
    def filter(self, record: LogRecord) -> bool:
        """Restores the level check for propagated records which Python skips by default."""
        return record.levelno >= getLogger(record.name).getEffectiveLevel()


class CustomHandler(RichHandler):
    def format(self, record: LogRecord) -> str:
        if record.args and record.name.startswith("vsview"):
            record.args = tuple(arg() if _is_lambda(arg) else arg for arg in record.args)
        return super().format(record)


class ThreadAwareFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        fmt = "{message}" if record.name.startswith("vsview") else "{name}: {message}"

        if record.threadName != main_thread_name:
            fmt = f"[{record.threadName}]: {fmt}"

        self._style._fmt = fmt
        return super().format(record)


def setup_logging(
    level: int | None = None,
    vs_level: int | None = None,
    vsview_level: int | None = None,
    vsengine_level: int | None = INFO,
    qt_level: int | None = None,
    capture_warnings: bool = True,
) -> None:
    qInstallMessageHandler(_qt_message_handler)

    level = fallback(level, INFO)

    # One handler to rule them all
    handler = CustomHandler(
        console=console,
        rich_tracebacks=True,
        log_time_format=lambda dt: Text("[{}.{:03d}]".format(dt.strftime("%H:%M:%S"), dt.microsecond // 1000)),
    )
    handler.setFormatter(ThreadAwareFormatter(style="{"))
    handler.addFilter(EffectiveLevelFilter())

    root_logger = getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # Set levels for specialized loggers—they will all propagate to the root handler
    getLogger("vapoursynth").setLevel(fallback(vs_level, level))
    getLogger("vsview").setLevel(fallback(vsview_level, level))
    getLogger("vsengine").setLevel(fallback(vsengine_level, level))
    getLogger("qt").setLevel(fallback(qt_level, level))

    if capture_warnings:
        import warnings

        warnings.filterwarnings("always")
        captureWarnings(True)
