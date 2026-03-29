from __future__ import annotations

import importlib.metadata
import random
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from functools import cache
from logging import getLogger
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
