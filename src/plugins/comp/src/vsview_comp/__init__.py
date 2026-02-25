from typing import Any

from vsview.api import WidgetPluginBase, hookimpl

from .main import CompPlugin


@hookimpl
def vsview_register_toolpanel() -> type[WidgetPluginBase[Any, Any]]:
    return CompPlugin
