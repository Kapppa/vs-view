from __future__ import annotations

import importlib
import importlib.metadata
import sys
import sysconfig
from collections.abc import Iterable
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)

EXCLUDED_PREFIXES = frozenset(
    {
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path(sys.exec_prefix).resolve(),
        Path(sys.base_exec_prefix).resolve(),
        Path(sysconfig.get_path("purelib")).resolve(),
        Path(sysconfig.get_path("platlib")).resolve(),
    }
)


def _get_installed_top_levels() -> frozenset[str]:
    return frozenset(
        importlib.metadata.packages_distributions().keys()
        | {dist.metadata["Name"].replace("-", "_") for dist in importlib.metadata.distributions()}
    )


def find_local_packages() -> set[str]:
    """Find top-level package names of locally-available (non-installed) modules."""
    local_packages = set[str]()
    installed = _get_installed_top_levels()

    for module in sys.modules.values():
        if not (mod_file := getattr(module, "__file__", None)):
            continue

        top_level = module.__name__.split(".")[0]

        if top_level in installed:
            continue

        mod_path = Path(mod_file).resolve()

        if any(mod_path.is_relative_to(p) for p in EXCLUDED_PREFIXES) or not mod_path.is_file():
            continue

        local_packages.add(top_level)

    return local_packages


def evict_packages(packages: Iterable[str]) -> None:
    """Evict all submodules of the given top-level packages from ``sys.modules``."""

    for package in sorted(packages):
        submodules = sorted(k for k in sys.modules if k == package or k.startswith(f"{package}."))

        for mod_name in reversed(submodules):
            del sys.modules[mod_name]

        logger.debug('Evicted package: "%s"', package)
