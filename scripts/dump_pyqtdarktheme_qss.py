import pathlib
import sys

import qdarktheme  # type: ignore
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)

# Dump dark theme
dark_stylesheet = qdarktheme.load_stylesheet("dark")
pathlib.Path("pyqtdarktheme_dark.qss").write_text(dark_stylesheet)
print("Saved: pyqtdarktheme_dark.qss")

# Dump light theme
light_stylesheet = qdarktheme.load_stylesheet("light")
pathlib.Path("pyqtdarktheme_light.qss").write_text(light_stylesheet)
print("Saved: pyqtdarktheme_light.qss")
