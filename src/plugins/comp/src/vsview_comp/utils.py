import random
from collections.abc import Sequence

from ._version import __version__


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
