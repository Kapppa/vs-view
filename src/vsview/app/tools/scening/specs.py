from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pluggy

if TYPE_CHECKING:
    from .api import Parser, Serializer

hookspec = pluggy.HookspecMarker("vsview.scening")
hookimpl = pluggy.HookimplMarker("vsview.scening")


@hookspec
def vsview_scening_register_parser() -> Parser | Sequence[Parser]:
    raise NotImplementedError


@hookspec
def vsview_scening_register_serializer() -> Serializer | Sequence[Serializer]:
    raise NotImplementedError
