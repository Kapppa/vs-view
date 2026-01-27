"""
Plugin hook specifications for vsview.

This module defines the interfaces that plugins should implement to extend the application's functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pluggy

if TYPE_CHECKING:
    from .api import WidgetPluginBase


hookspec = pluggy.HookspecMarker("vsview")
"""Marker to be used for hook specifications."""

hookimpl = pluggy.HookimplMarker("vsview")
"""Marker to be used for hook implementations."""


# UI Hooks
@hookspec
def vsview_register_tooldock() -> type[WidgetPluginBase[Any, Any]]:
    """
    Register a tool dock widget.

    Returns:
        A WidgetPluginBase subclass defining a QDockWidget-based tool.
    """
    raise NotImplementedError


@hookspec
def vsview_register_toolpanel() -> type[WidgetPluginBase[Any, Any]]:
    """
    Register a tool panel widget.

    Returns:
        A WidgetPluginBase subclass defining a panel-based tool.
    """
    raise NotImplementedError
    raise NotImplementedError
