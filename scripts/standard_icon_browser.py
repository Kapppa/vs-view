import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication, QHeaderView, QStyle, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class StandardIconBrowser(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PySide6 Standard Pixmaps")
        self.resize(450, 600)

        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Icon", "Enum Name"])

        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.table.setIconSize(QSize(32, 32))

        layout.addWidget(self.table)

        self.load_icons()

    def load_icons(self) -> None:
        style = self.style()

        for member in QStyle.StandardPixmap:
            if member.name == "SP_CustomBase":
                continue

            row_idx = self.table.rowCount()
            self.table.insertRow(row_idx)

            icon = style.standardIcon(member)

            icon_item = QTableWidgetItem(icon, "")
            name_item = QTableWidgetItem(member.name)
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

            self.table.setItem(row_idx, 0, icon_item)
            self.table.setItem(row_idx, 1, name_item)

            self.table.setRowHeight(row_idx, 40)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = StandardIconBrowser()
    window.show()

    sys.exit(app.exec())
