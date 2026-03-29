from typing import TYPE_CHECKING, Any

__all__ = ["catch_output", "is_preview", "set_output"]

if TYPE_CHECKING:
    from .api import catch_output, is_preview, set_output
else:

    def __getattr__(name: str) -> Any:
        from importlib import import_module

        if name in __all__ and (attr := getattr(import_module("vsview.api"), name, None)):
            return attr

        raise AttributeError(f"Cannot import {name!r} from 'vsview'")
