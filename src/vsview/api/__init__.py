"""API for vsview"""

from ..app.icon import IconName, IconReloadMixin, load_icon
from ..app.packing import Packer, get_packer
from ..app.plugins import (
    AudioOutputProxy,
    GraphicsViewProxy,
    LocalSettingsModel,
    NodeProcessor,
    PluginAPI,
    PluginGraphicsView,
    PluginSecrets,
    PluginSettings,
    VideoOutputProxy,
    WidgetPluginBase,
    hookimpl,
)
from ..app.settings.action import ActionDefinition
from ..app.settings.metadata import (
    Checkbox,
    ColorPicker,
    DoubleSpin,
    Dropdown,
    FilePicker,
    LineEdit,
    ListEdit,
    Login,
    PlainTextEdit,
    Spin,
    WidgetMetadata,
    WidgetTimeEdit,
)
from ..app.settings.widgets import (
    ColorPickerInput,
    FilePickerWidget,
    ListEditWidget,
    LoginCredentialsInput,
)
from ..app.views import OutputInfo
from ..app.views.components import AbstractTableModel, Accordion, AnimatedToggle, NonClosingMenu, SegmentedControl
from ..app.views.timeline import FrameEdit, TimeEdit
from ..app.views.video import BaseGraphicsView
from ..types import Frame, Time
from ..vsenv import run_in_background, run_in_loop
from .info import is_preview
from .output import catch_output, set_output

__all__ = [
    "AbstractTableModel",
    "Accordion",
    "ActionDefinition",
    "AnimatedToggle",
    "AudioOutputProxy",
    "BaseGraphicsView",
    "Checkbox",
    "ColorPicker",
    "ColorPickerInput",
    "DoubleSpin",
    "Dropdown",
    "FilePicker",
    "FilePickerWidget",
    "Frame",
    "FrameEdit",
    "GraphicsViewProxy",
    "IconName",
    "IconReloadMixin",
    "LineEdit",
    "ListEdit",
    "ListEditWidget",
    "LocalSettingsModel",
    "Login",
    "LoginCredentialsInput",
    "NodeProcessor",
    "NonClosingMenu",
    "OutputInfo",
    "Packer",
    "PlainTextEdit",
    "PluginAPI",
    "PluginGraphicsView",
    "PluginSecrets",
    "PluginSettings",
    "SegmentedControl",
    "Spin",
    "Time",
    "TimeEdit",
    "VideoOutputProxy",
    "WidgetMetadata",
    "WidgetPluginBase",
    "WidgetTimeEdit",
    "catch_output",
    "get_packer",
    "hookimpl",
    "is_preview",
    "load_icon",
    "run_in_background",
    "run_in_loop",
    "set_output",
]
