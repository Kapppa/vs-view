import pathlib
import sys

import qdarkstyle  # type: ignore
from PySide6.QtWidgets import QApplication
from qdarkstyle.dark.palette import DarkPalette  # type: ignore
from qdarkstyle.light.palette import LightPalette  # type: ignore

app = QApplication(sys.argv)

# Dump dark theme
dark_stylesheet = qdarkstyle.load_stylesheet(palette=DarkPalette)
pathlib.Path("qdarkstyle_dark.qss").write_text(dark_stylesheet)
print("Saved: qdarkstyle_dark.qss")

# Dump light theme
light_stylesheet = qdarkstyle.load_stylesheet(palette=LightPalette)
pathlib.Path("qdarkstyle_light.qss").write_text(light_stylesheet)
print("Saved: qdarkstyle_light.qss")
