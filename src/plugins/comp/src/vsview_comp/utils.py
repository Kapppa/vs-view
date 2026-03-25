from __future__ import annotations

import importlib.metadata
import random
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar
from functools import cache, wraps
from http.cookiejar import CookieJar
from inspect import iscoroutinefunction
from logging import DEBUG, INFO, LogRecord, getLogger
from types import TracebackType

import niquests

logger = getLogger(__name__)


@cache
def get_slowpics_headers() -> dict[str, str]:
    version = importlib.metadata.version("vsview-comp")
    return {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9",
        "Access-Control-Allow-Origin": "*",
        "Origin": "https://slow.pics/",
        "Referer": "https://slow.pics/comparison",
        "User-Agent": (
            f"vs-view (https://github.com/Jaded-Encoding-Thaumaturgy/vs-view {version})"  # SlowBro asked for this
        ),
    }


_rev_conf = niquests.RevocationConfiguration(niquests.RevocationStrategy.PREFER_CRL)

_demote_niquests_ctx = ContextVar("_demote_niquests_ctx", default=False)


def niquests_demote_filter(record: LogRecord) -> bool:
    if _demote_niquests_ctx.get() and record.levelno == INFO:
        record.levelno = DEBUG
        record.levelname = "DEBUG"
    return True


_niquests_logger = getLogger("niquests")
_niquests_logger.addFilter(niquests_demote_filter)


def demote_niquests_logs[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    if iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            token = _demote_niquests_ctx.set(True)
            try:
                return await func(*args, **kwargs)
            finally:
                _demote_niquests_ctx.reset(token)

        return async_wrapper  # type: ignore[return-value]

    @wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        token = _demote_niquests_ctx.set(True)
        try:
            return func(*args, **kwargs)
        finally:
            _demote_niquests_ctx.reset(token)

    return sync_wrapper


class LogNiquestsErrors(AbstractContextManager[None], AbstractAsyncContextManager[None]):
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
        if isinstance(exc_val, niquests.HTTPError):
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


def get_cookie(jar: CookieJar, name: str) -> str | None:
    for cookie in jar:
        if cookie.name == name:
            return cookie.value
    return ""


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
