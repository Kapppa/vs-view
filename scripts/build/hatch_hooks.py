from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface[Any]):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        sbom_path = Path(self.root) / "sboms.json"
        if sbom_path.exists():
            build_data.setdefault("sbom_files", []).append(str(sbom_path.relative_to(self.root)))
