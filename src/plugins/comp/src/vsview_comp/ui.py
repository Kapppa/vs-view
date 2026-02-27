from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple, Self

import jinja2
from jetpytools import cachedproperty
from PySide6.QtCore import QBuffer, QCoreApplication, QEvent, QIODevice, QObject, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QKeyEvent, QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
from shiboken6 import Shiboken
from vapoursynth import RGB24, VideoNode
from vstools import core, get_prop

from vsview.api import NonClosingMenu, PluginAPI, Time, VideoOutputProxy, run_in_background, run_in_loop

from .models import TMDBTitle


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


class FrameSourceProvider(StrEnum):
    MANUAL = "Manual"
    RANDOM = "Random"
    RANDOM_DARK = "Random dark"
    RANDOM_LIGHT = "Random light"


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

    def add_item(
        self,
        frame: int | None = None,
        get_pict_type: bool = False,
        src_provider: FrameSourceProvider = FrameSourceProvider.MANUAL,
    ) -> None:
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
        item.setToolTip(f'{item.text()} from "{src_provider}"')

        # Find sorted insertion position
        insert_idx = self.count()
        for i in range(insert_idx):
            if self.item(i).data(self.TIME_ROLE) > time:
                insert_idx = i
                break

        self.insertItem(insert_idx, item)
        self.scrollToItem(item)
        self.listSizeChanged.emit(1)

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

    @run_in_loop(return_future=False)
    def update_item_icon(self, item: QListWidgetItem, image: QImage, pict_type: str) -> None:
        with suppress(RuntimeError):
            item.setIcon(QPixmap.fromImage(image))
            item.setData(self.PICT_TYPE_ROLE, pict_type)

            if pict_type != "?":
                item.setText(item.text() + f"({pict_type})")

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


class PushButton(QPushButton):
    enabledChanged = Signal(bool)

    def changeEvent(self, e: QEvent) -> None:
        if e.type() == QEvent.Type.EnabledChange:
            self.enabledChanged.emit(self.isEnabled())

        super().changeEvent(e)


class PosterPayload(NamedTuple):
    result_serial: int
    row: int
    url: str


class TMDBListPopup(QListWidget):
    POSTER_URL_PREFIX = "https://image.tmdb.org/t/p/w92"

    def __init__(self, parent: QWidget, target: QLineEdit) -> None:
        super().__init__(parent)
        if not (app := QCoreApplication.instance()):
            raise SystemError

        self._app = app
        self._target: QLineEdit | None = target
        self._target_window: QWidget | None = target.window()
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setIconSize(QSize(32, 45))
        self.setMaximumHeight(300)
        self.hide()

        # Incremented on each new result set. Used to ignore stale async replies.
        self._results_serial = 0
        self._poster_cache = dict[str, QIcon]()
        self._poster_replies = dict[QNetworkReply, PosterPayload]()
        self._poster_loader = QNetworkAccessManager(self)
        self._poster_loader.finished.connect(self._on_poster_downloaded)

        self._app.installEventFilter(self)

        self._target.installEventFilter(self)
        self._target_window.installEventFilter(self)

        self._target.destroyed.connect(self._on_target_destroyed)
        self.destroyed.connect(self._cleanup_event_filters)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._app and event.type() == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
            self._handle_app_click(event.globalPosition().toPoint())

        if watched in [self._safe_target(), self._target_window]:
            if event.type() in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show) and self.isVisible():
                self.update_geometry()
            elif event.type() == QEvent.Type.Hide:
                self.hide()

        return super().eventFilter(watched, event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            delta = event.angleDelta().y() or event.angleDelta().x()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta)
            event.accept()
        else:
            super().wheelEvent(event)

    @cachedproperty
    def tool_tip_template(self) -> jinja2.Template:
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent),
            autoescape=jinja2.select_autoescape(),
        ).get_template("tmdb_tooltip.html.jinja")

    def show_results(self, results: list[TMDBTitle]) -> None:
        self._results_serial += 1
        # Cancel old requests when query/results change.
        self._cancel_poster_requests()
        self.clear()

        for row, value in enumerate(results):
            item = QListWidgetItem(value.name)
            item.setToolTip(self._get_tmdb_tooltip(value))
            item.setData(Qt.ItemDataRole.UserRole, value)
            self.addItem(item)

            if not value.poster_path:
                continue

            url = f"{self.POSTER_URL_PREFIX}{value.poster_path}"
            if icon := self._poster_cache.get(url):
                item.setIcon(icon)
                continue

            # Load each poster asynchronously so text results appear instantly.
            reply = self._poster_loader.get(QNetworkRequest(url))
            self._poster_replies[reply] = PosterPayload(self._results_serial, row, url)

        self.setCurrentRow(0)
        self.update_geometry()
        self.show()

    def show_no_results(self) -> None:
        self._results_serial += 1
        self._cancel_poster_requests()
        self.clear()
        item = QListWidgetItem("No results found")
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setData(Qt.ItemDataRole.UserRole, None)
        self.addItem(item)
        self.clearSelection()
        self.setCurrentRow(-1)
        self.update_geometry()
        self.show()

    def update_geometry(self) -> None:
        if not (target := self._safe_target()):
            self.hide()
            return

        pos = target.mapToGlobal(QPoint(0, target.height()))
        self.setFixedWidth(target.width())
        self.move(pos)

    # Poster network
    def _cancel_poster_requests(self) -> None:
        if not self._poster_replies:
            return

        pending = list(self._poster_replies)
        self._poster_replies.clear()

        for reply in pending:
            reply.abort()
            reply.deleteLater()

    def _on_poster_downloaded(self, reply: QNetworkReply) -> None:
        if not (payload := self._poster_replies.pop(reply, None)):
            reply.deleteLater()
            return

        # Ignore responses from older result sets or failed downloads.
        if payload.result_serial != self._results_serial or reply.error() != QNetworkReply.NetworkError.NoError:
            reply.deleteLater()
            return

        pixmap = QPixmap()
        pixmap.loadFromData(reply.readAll())

        if pixmap.isNull():
            reply.deleteLater()
            return

        icon = QIcon(
            pixmap.scaled(
                self.iconSize(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._poster_cache[payload.url] = icon

        if payload.row < self.count():
            item = self.item(payload.row)
            item.setIcon(icon)
            item.setToolTip(self._get_tmdb_tooltip(item.data(Qt.ItemDataRole.UserRole), pixmap_data_uri(pixmap)))

        reply.deleteLater()

    def _get_tmdb_tooltip(self, title: TMDBTitle, poster_data_uri: str | None = None) -> str:
        rows = [
            {"label": "Original title", "value": title.tooltip_data.original_name},
            {"label": "Genres", "value": title.tooltip_data.genres},
            {"label": "Language", "value": title.tooltip_data.language},
            {"label": "Country", "value": title.tooltip_data.country},
            {"label": "Release", "value": title.tooltip_data.release_date},
            {"label": "Rating", "value": title.tooltip_data.rating},
            {"label": "Popularity", "value": title.tooltip_data.popularity},
            {"label": "TMDB ID", "value": title.tooltip_data.tmdb_id},
        ]

        return self.tool_tip_template.render(
            header=title.tooltip_data.header,
            poster_data_uri=poster_data_uri,
            meta_rows=rows,
            overview=title.tooltip_data.overview,
        )

    # App clicks
    def _handle_app_click(self, global_pos: QPoint) -> None:
        if not self.isVisible():
            return

        clicked = QApplication.widgetAt(global_pos)
        target = self._safe_target()

        is_inside_popup = clicked is self or (clicked and self.isAncestorOf(clicked))
        is_inside_target = clicked and target and (clicked is target or target.isAncestorOf(clicked))

        if not (is_inside_popup or is_inside_target):
            self.hide()

    def _safe_target(self) -> QLineEdit | None:
        if self._target and Shiboken.isValid(self._target):
            return self._target

        self._target = None
        return None

    def _on_target_destroyed(self) -> None:
        self._target = None
        self._target_window = None
        self.hide()

    def _cleanup_event_filters(self, *_: Any) -> None:
        self._cancel_poster_requests()

        if Shiboken.isValid(self._app):
            self._app.removeEventFilter(self)

        if target := self._safe_target():
            target.removeEventFilter(self)

        if self._target_window and Shiboken.isValid(self._target_window):
            self._target_window.removeEventFilter(self)


@contextmanager
def open_qbuffer(mode: QIODevice.OpenModeFlag = QIODevice.OpenModeFlag.WriteOnly) -> Iterator[QBuffer]:
    buffer = QBuffer()
    buffer.open(mode)
    try:
        yield buffer
    finally:
        buffer.close()


def pixmap_data_uri(pixmap: QPixmap) -> str:
    with open_qbuffer() as buffer:
        pixmap.save(buffer, "PNG", 100)

    b64_str = str(buffer.data().toBase64(), "ascii")  # type: ignore[call-overload]

    return f"data:image/png;base64,{b64_str}"
