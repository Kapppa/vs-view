from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar
from functools import wraps
from inspect import iscoroutinefunction
from logging import DEBUG, INFO, LogRecord, getLogger
from types import TracebackType

import httpx

from ._version import __version__

logger = getLogger(__name__)


def get_slowpics_headers() -> dict[str, str]:
    return {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9",
        "Access-Control-Allow-Origin": "*",
        "Origin": "https://slow.pics/",
        "Referer": "https://slow.pics/comparison",
        "User-Agent": (
            f"vs-view (https://github.com/Jaded-Encoding-Thaumaturgy/vs-view {__version__})"  # SlowBro asked for this
        ),
    }


_demote_httpx_ctx = ContextVar("_demote_httpx_ctx", default=False)


def httpx_demote_filter(record: LogRecord) -> bool:
    if _demote_httpx_ctx.get() and record.levelno == INFO:
        record.levelno = DEBUG
        record.levelname = "DEBUG"
    return True


_httpx_logger = getLogger("httpx")
_httpx_logger.addFilter(httpx_demote_filter)


def demote_httpx_logs[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    if iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            token = _demote_httpx_ctx.set(True)
            try:
                return await func(*args, **kwargs)
            finally:
                _demote_httpx_ctx.reset(token)

        return async_wrapper  # type: ignore[return-value]

    @wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        token = _demote_httpx_ctx.set(True)
        try:
            return func(*args, **kwargs)
        finally:
            _demote_httpx_ctx.reset(token)

    return sync_wrapper


class LogHTTPXErrors(AbstractContextManager[None], AbstractAsyncContextManager[None]):
    def __init__(self, ctx_message: str) -> None:
        self.ctx_message = ctx_message

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_t: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        if isinstance(exc_val, httpx.HTTPError):
            logger.error("%s failed: %s", self.ctx_message, exc_val, stacklevel=4)
            logger.debug("Full traceback", exc_info=exc_val, stacklevel=4)
            return True
        return None

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_t: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return self.__exit__(exc_t, exc_val, tb)


def get_random_number_interval(min_val: int, max_val: int, count: int, index: int, exclude: Sequence[int]) -> int:
    """Picks a random, non-excluded number from a specific subset of a range."""
    if not (0 <= index < count):
        raise ValueError(f"{index} is out of range of 0-{count - 1}")

    interval = (max_val - min_val) // count
    lo = min_val + interval * index
    hi = min_val + interval * (index + 1)

    pool_size = hi - lo + 1

    for _ in range(pool_size):
        if (rnum := random.randrange(lo, hi)) not in exclude:
            return rnum

    raise ValueError(f"All {pool_size} values in interval [{lo}, {hi}] are excluded")
