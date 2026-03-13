import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFrame, QGridLayout, QLabel, QMainWindow, QVBoxLayout, QWidget


class FrameDemo(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("QFrame Shape and Shadow Demo")
        self.resize(600, 700)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QGridLayout(central_widget)
        layout.setSpacing(15)

        # Column headers (Shadows)
        for col, shadow in enumerate(QFrame.Shadow):
            header = QLabel(f"<b>{shadow.name}</b>", self)
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(header, 0, col + 1)

        # Generate the grid
        for row, shape in enumerate(QFrame.Shape):
            # Row header (Shapes)
            row_header = QLabel(f"<b>{shape.name}</b>", self)
            layout.addWidget(row_header, row + 1, 0)

            for col, shadow in enumerate(QFrame.Shadow):
                frame = QFrame(self, frameShape=shape, frameShadow=shadow, lineWidth=3, midLineWidth=2)
                frame.setFixedSize(120, 80)

                # Add a label inside the frame (except for simple lines)
                if shape.name not in ["HLine", "VLine", "NoFrame"]:
                    inner_layout = QVBoxLayout()
                    inner_label = QLabel(
                        f"{shape.name}\n+\n{shadow.name}",
                        self,
                        alignment=Qt.AlignmentFlag.AlignCenter,
                    )
                    font = inner_label.font()
                    font.setPointSize(8)
                    inner_label.setFont(font)

                    inner_layout.addWidget(inner_label)
                    frame.setLayout(inner_layout)

                layout.addWidget(frame, row + 1, col + 1, Qt.AlignmentFlag.AlignCenter)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set a specific style to see how OS themes affect QFrames
    # app.setStyle("Fusion")

    window = FrameDemo()
    window.show()
    sys.exit(app.exec())
