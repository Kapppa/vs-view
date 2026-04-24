import os
from logging import getLogger

import dotenv

_TRUTHY = frozenset({"1", "true", "yes", "on"})

logger = getLogger(__name__)


def getenv_bool(key: str, default: bool = False) -> bool:
    return default if (val := os.getenv(key)) is None else val.strip().lower() in _TRUTHY


def load_dotenv() -> None:
    dotenv_path = dotenv.find_dotenv(usecwd=True)

    if not dotenv_path:
        return

    logger.info("Loading .env file from %s", dotenv_path)
    dotenv.load_dotenv(dotenv_path)
