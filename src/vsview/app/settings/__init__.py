"""Settings submodule for vsview."""

from .manager import SettingsManager
from .models import ActionID
from .secrets import SecretsError, SecretsManager
from .shortcuts import ShortcutManager

__all__ = ["ActionID", "SecretsError", "SecretsManager", "SettingsManager", "ShortcutManager"]
