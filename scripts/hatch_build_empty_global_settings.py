import json
import os
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface[Any]):
    GLOBAL_SETTINGS_PATH = Path("src/vsview/global_settings.json")

    @property
    def is_ci(self) -> bool:
        return os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true"

    def clean(self, versions: list[str]) -> None:
        self.GLOBAL_SETTINGS_PATH.unlink(missing_ok=True)

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.is_ci:
            self.GLOBAL_SETTINGS_PATH.write_text(json.dumps({}))

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        if self.is_ci:
            self.GLOBAL_SETTINGS_PATH.unlink(missing_ok=True)
