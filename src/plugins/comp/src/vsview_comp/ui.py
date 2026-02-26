from __future__ import annotations

from contextlib import suppress
from typing import Self

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from vapoursynth import RGB24, VideoNode, core
from vstools import get_prop

from vsview.api import NonClosingMenu, PluginAPI, Time, VideoOutputProxy, run_in_background, run_in_loop


class MainCompWidget(QScrollArea):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QScrollArea.Shape.NoFrame)

        self.container = QWidget(self)
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(8, 8, 8, 8)
        self.container_layout.setSpacing(8)

        self.setWidget(self.container)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.finalize()

    def add_section(self, section: QWidget) -> None:
        self.container_layout.addWidget(section)

    def finalize(self) -> None:
        self.container_layout.addStretch(1)


class PaddedCheckBox(QWidget):
    toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Include this output")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.checkbox = QCheckBox(self)
        self.checkbox.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.checkbox.toggled.connect(self.toggled.emit)
        layout.addWidget(self.checkbox)

    def setChecked(self, checked: bool) -> None:
        self.checkbox.setChecked(checked)

    def isChecked(self) -> bool:
        return self.checkbox.isChecked()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.checkbox.toggle()
            event.accept()
            return

        super().mouseReleaseEvent(event)


class OutputItemWidget(QWidget):
    inclusionChanged = Signal(bool)
    nameChanged = Signal(str)

    def __init__(self, vout: VideoOutputProxy, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.included = True
        self.vs_index = vout.vs_index
        self.vs_name = vout.vs_name
        self.num_frames = vout.vs_output.clip.num_frames
        self.duration = vout.info.total_duration

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)

        self.checkbox = PaddedCheckBox(self)
        self.checkbox.setChecked(True)
        self.checkbox.toggled.connect(self._on_check)

        self.name_edit = QLineEdit(self.vs_name, self)
        self.name_edit.textChanged.connect(self._on_name)

        self.info_label = QLabel(f"{self.duration.to_ts('{H:01d}:{M:02d}:{S:02d}.{ms:03d}')} ({self.num_frames})", self)

        layout.addWidget(self.checkbox)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.info_label)

    # Accept all clicks so they don't propagate to the menu and close it.
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        event.accept()

    def _on_check(self, state: bool) -> None:
        self.included = state
        self.inclusionChanged.emit(state)

    def _on_name(self, text: str) -> None:
        self.vs_name = text
        self.nameChanged.emit(text)


class OutputDropdown(QPushButton):
    inclusionChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.items = list[OutputItemWidget]()
        self.total_outputs = 0
        self.shortest_dur_text = ""

        self.setMenu(NonClosingMenu(self))

        self.setStyleSheet("text-align: left; padding: 4px;")

    @property
    def included_outputs(self) -> list[int]:
        return [w.vs_index for w in self.items if w.included]

    def populate(self, voutputs: list[VideoOutputProxy]) -> None:
        self.menu().clear()
        self.items.clear()
        self.total_outputs = len(voutputs)

        shortest = min(voutputs, key=lambda v: v.info.total_duration)
        dur = shortest.info.total_duration.to_ts("{H:01d}:{M:02d}:{S:02d}.{ms:03d}")
        frames = shortest.vs_output.clip.num_frames
        self.shortest_dur_text = f" - {dur} ({frames})"

        for vout in voutputs:
            action = QWidgetAction(self.menu())
            action.setCheckable(True)

            widget = OutputItemWidget(vout, self.menu())
            action.setDefaultWidget(widget)
            self.menu().addAction(action)
            self.items.append(widget)

            widget.inclusionChanged.connect(self._update_text)
            widget.nameChanged.connect(self._update_text)

        self._update_text()

    def _update_text(self) -> None:
        included = [w for w in self.items if w.included]

        if len(included) == self.total_outputs and self.total_outputs > 0:
            self.setText(f"All outputs{self.shortest_dur_text}")
        elif len(included) == 0:
            self.setText(f"No output{self.shortest_dur_text}")
        else:
            names = [w.vs_name for w in included]
            self.setText(f"{', '.join(names)}{self.shortest_dur_text}")

        self.inclusionChanged.emit()


class FrameThumbnailList(QListWidget):
    ICON_SIZE = QSize(112, 63)
    TIME_ROLE = Qt.ItemDataRole.UserRole
    PICT_TYPE_ROLE = Qt.ItemDataRole.UserRole + 1

    listSizeChanged = Signal(int)  # delta

    def __init__(self, api: PluginAPI, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.api = api
        self.clip_cache = dict[int, VideoNode]()

        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(False)
        self.setIconSize(self.ICON_SIZE)
        self.setFixedHeight(116)
        self.setSpacing(4)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("QListWidget { padding-right: 12px; }")

        self.setMovement(QListWidget.Movement.Static)
        self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)

        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        self.api.register_on_destroy(self.clip_cache.clear)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete:
            self.remove_selected()
            event.accept()
        else:
            super().keyPressEvent(event)

    @property
    def thumbnail_clip(self) -> VideoNode:
        voutput = self.api.current_voutput

        if voutput.vs_index not in self.clip_cache:
            ar = voutput.vs_output.clip.width / voutput.vs_output.clip.height

            target_h = self.ICON_SIZE.height()
            target_w = round(target_h * ar)

            if target_w > self.ICON_SIZE.width():
                target_w = self.ICON_SIZE.width()
                target_h = round(target_w * 1 / ar)

            downscaled = core.resize.Bilinear(
                voutput.vs_output.clip,
                target_w,
                target_h,
                format=RGB24,
            )
            packed = self.api.packer.to_rgb_packed(downscaled, alpha=None)

            self.clip_cache[voutput.vs_index] = packed

        return self.clip_cache[voutput.vs_index]

    def add_item(self, frame: int | None = None, get_pict_type: bool = False) -> None:
        if frame is not None:
            time = self.api.current_voutput.frame_to_time(frame)
        else:
            time = self.api.current_time
            frame = self.api.current_frame

        # Reject duplicates
        for i in range(self.count()):
            if self.item(i).data(self.TIME_ROLE) == time:
                self.setCurrentRow(i)
                return

        item = QListWidgetItem(f"{time.to_ts('{H:01d}:{M:02d}:{S:02d}.{ms:03d}')} ({frame})")
        item.setData(self.TIME_ROLE, time)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        # Ensure the item has space for the icon before it's loaded
        item.setSizeHint(QSize(self.ICON_SIZE.width() + self.spacing() * 2, 92))

        # Find sorted insertion position
        insert_idx = self.count()
        for i in range(insert_idx):
            if self.item(i).data(self.TIME_ROLE) > time:
                insert_idx = i
                break

        self.insertItem(insert_idx, item)
        self.scrollToItem(item)

        self.fetch_thumbnail(frame, item, get_pict_type)

    def remove_selected(self) -> None:
        nb = len(self.selectedItems())

        for item in self.selectedItems():
            self.takeItem(self.row(item))

        self.listSizeChanged.emit(-nb)

    def get_data(self) -> list[tuple[Time, str]]:
        return [
            (self.item(i).data(self.TIME_ROLE), self.item(i).data(self.PICT_TYPE_ROLE)) for i in range(self.count())
        ]

    @run_in_background(name="FetchThumbnail")
    def fetch_thumbnail(self, n: int, item: QListWidgetItem, get_pict_type: bool) -> None:
        with self.api.vs_context(), self.thumbnail_clip.get_frame(n) as f:
            self.update_item_icon(
                item,
                self.api.packer.frame_to_qimage(f).copy(),
                get_prop(f, "_PictType", str, default="?", func=self.fetch_thumbnail) if get_pict_type else "?",
            )

    @run_in_loop
    def update_item_icon(self, item: QListWidgetItem, image: QImage, pict_type: str) -> None:
        with suppress(RuntimeError):
            item.setIcon(QPixmap.fromImage(image))
            item.setData(self.PICT_TYPE_ROLE, pict_type)
            if pict_type != "?":
                item.setText(item.text() + f"({pict_type})")
            self.listSizeChanged.emit(1)

    def show_context_menu(self, pos: QPoint) -> None:
        if not self.itemAt(pos):
            return

        menu = QMenu(self)

        remove_action = menu.addAction(
            f"Remove {count} frame(s)" if (count := len(self.selectedItems())) > 1 else "Remove frame"
        )
        remove_action.triggered.connect(self.remove_selected)

        menu.exec(self.mapToGlobal(pos))
        menu.deleteLater()


class ProgressBar(QProgressBar):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(24)
        self.setRange(0, 100)
        self.setTextVisible(True)
        self.setToolTip("Overall progress of extraction and upload tasks")

    @run_in_loop(return_future=False)
    def update_progress(
        self,
        *,
        value: int | None = None,
        range: tuple[int, int] | None = None,
        fmt: str | None = None,
        increment: int | None = None,
    ) -> None:
        if range:
            self.setRange(*range)
        if fmt:
            self.setFormat(fmt)
        if value is not None:
            self.setValue(value)
        if increment is not None:
            self.setValue(self.value() + increment)

    @run_in_loop(return_future=False)
    def reset_progress(self) -> None:
        self.reset()
        self.setFormat("%p%")
