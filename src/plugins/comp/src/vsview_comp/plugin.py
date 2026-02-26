import ast
from collections.abc import Sequence
from concurrent.futures import Future, wait
from datetime import datetime
from functools import cache
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any

from jetpytools import clamp, ndigits
from pathvalidate import sanitize_filepath
from pydantic import BaseModel
from PySide6.QtCore import QObject, Qt, QTime
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from vapoursynth import GRAY8, RGB24, VideoNode
from vstools import clip_async_render, clip_data_gather, core, get_prop, remap_frames

from vsview.api import (
    Accordion,
    ActionDefinition,
    Frame,
    FrameEdit,
    IconName,
    IconReloadMixin,
    LineEdit,
    Login,
    PluginAPI,
    SegmentedControl,
    Time,
    TimeEdit,
    VideoOutputProxy,
    WidgetPluginBase,
    run_in_background,
    run_in_loop,
)

from .ui import FrameSourceProvider, FrameThumbnailList, MainCompWidget, OutputDropdown, ProgressBar
from .utils import get_random_number_interval

logger = getLogger(__name__)


PLUGIN_ID = "jet_vsview_comp"
PLUGIN_DISPLAY = "Comparison"
COOKIE_KEY = "cookies.v1"


class GlobalSettings(BaseModel):
    login: Annotated[str, Login(label="Slow.pics credentials", namespace=PLUGIN_ID)]

    # tmdb_movie_format: Annotated[
    #     str,
    #     LineEdit("Format to use when selecting a Movie from TMDB"),
    # ] = "{tmdb_title} ({tmdb_year}) - {video_nodes}"
    # tmdb_tv_format: Annotated[
    #     str,
    #     LineEdit("Format to use when selecting a TV Show from TMDB"),
    # ] = "{tmdb_title} ({tmdb_year}) - S01E01 - {video_nodes}"
    # open_comp_automatically: Annotated[
    #     bool,
    #     Checkbox(
    #         label="Open comp links automatically",
    #         text="",
    #         tooltip="Will open the link to the comp once it has finished automatically.",
    #     ),
    # ] = False

    pict_types_i: bool = True
    pict_types_p: bool = True
    pict_types_b: bool = True
    combed: bool = False
    public_comp_default: bool = True


class CompPlugin(WidgetPluginBase[GlobalSettings, None], IconReloadMixin):
    identifier = PLUGIN_ID
    display_name = PLUGIN_DISPLAY

    shortcuts = (ActionDefinition(f"{PLUGIN_ID}.add_current_frame", "Add Current Frame", "Ctrl+Space"),)

    def __init__(self, parent: QWidget, api: PluginAPI) -> None:
        super().__init__(parent, api)

        self._pending_select_frames: Future[Any] | None = None
        self._pending_extract_frames: Future[Any] | None = None

        self.pict_types_supported = True

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        with MainCompWidget(self) as main:
            self._setup_clip_options(main)
            self._setup_upload_settings(main)
            self._setup_actions(main)

        main_layout.addWidget(main)

        self.progress_bar = ProgressBar(self)
        main_layout.addWidget(self.progress_bar)

        # Disable the whole plugin if we don't have a local storage
        self.setEnabled(bool(self.api.file_path))

        self.api.register_action(
            f"{PLUGIN_ID}.add_current_frame",
            self.add_frame_act,
            context=Qt.ShortcutContext.WindowShortcut,
        )

        self.api.register_on_destroy(self.init_load.cache_clear)

    def _setup_clip_options(self, main: MainCompWidget) -> None:
        self.clip_section = Accordion("Clip Options", main)
        self.clip_section.setToolTip("Configure source clips and select frames for comparison")

        form = self.clip_section.add_form_layout()

        self.outputs_dropdown = OutputDropdown(self.clip_section)
        self.outputs_dropdown.setToolTip("Select and rename outputs.")
        self.outputs_dropdown.populate(self.api.voutputs)
        form.addRow(self.outputs_dropdown)

        # Current frames + add remove buttons
        frame_widget = QWidget(self.clip_section)
        frame_row = QVBoxLayout(frame_widget)
        frame_row.setContentsMargins(0, 0, 0, 0)
        frame_row.setSpacing(4)

        toolbar = QToolBar(
            frame_widget,
            movable=False,
            toolButtonStyle=Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )

        self.add_frame_act = self.make_action(IconName.PLUS, "Add current frame to the list", frame_widget)
        self.add_frame_act.setIconText("Add current frame")

        menu = QMenu(frame_widget)
        self.add_multi_frames_act = menu.addAction("Add multiple frames...")
        self.add_multi_frames_act.triggered.connect(self.on_add_multiple_frames)
        self.add_frame_act.setMenu(menu)

        toolbar.addAction(self.add_frame_act)
        if isinstance(add_btn := toolbar.widgetForAction(self.add_frame_act), QToolButton):
            add_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            add_btn.setStyleSheet("QToolButton::menu-button:hover {background-color: rgba(255, 255, 255, 0.1)}")

        self.remove_frame_act = self.make_action(IconName.MINUS, "Remove selected frame(s) from the list", frame_widget)
        self.remove_frame_act.setIconText("Remove selected frame(s)")
        self.remove_frame_act.setEnabled(False)

        toolbar.addAction(self.remove_frame_act)

        self.frames_list = FrameThumbnailList(self.api, frame_widget)
        self.frames_list.setToolTip("Double-click to seek to frame. Use 'Delete' to remove.")
        self.frames_list.itemDoubleClicked.connect(
            lambda item: self.api.playback.seek(
                self.api.current_voutput.time_to_frame(item.data(Qt.ItemDataRole.UserRole))
            )
        )
        self.frames_list.itemSelectionChanged.connect(
            lambda: self.remove_frame_act.setEnabled(len(self.frames_list.selectedItems()) > 0)
        )
        self.frames_list.listSizeChanged.connect(self.on_list_size_changed)

        self.add_frame_act.triggered.connect(lambda: self.frames_list.add_item())
        self.remove_frame_act.triggered.connect(self.frames_list.remove_selected)

        frame_row.addWidget(toolbar)
        frame_row.addWidget(self.frames_list)
        form.addRow(frame_widget)

        # Group box for the Auto Select automation
        auto_select_container = QGroupBox("Auto Select", self.clip_section)
        auto_select_container.setToolTip("Automated frame selection tools based on brightness and picture type")
        auto_select_frame_layout = QFormLayout(auto_select_container)

        # Frame counts
        frame_count_widget = QWidget(auto_select_container)
        frame_count_layout = QHBoxLayout(frame_count_widget)
        frame_count_layout.setContentsMargins(0, 0, 0, 0)
        frame_count_layout.setSpacing(8)

        self.random_frame_count = FrameEdit(frame_count_widget)
        self.random_frame_count.setRange(0, 40)  # 40 should be more than enough
        self.random_frame_count.setValue(0)
        self.random_frame_count.setFixedWidth(80)
        self.random_frame_count.setToolTip("Total number of random frames to select")
        self.random_frame_count.frameChanged.connect(self.on_random_frame_count_changed)

        self.dark_frame_count = FrameEdit(frame_count_widget)
        self.dark_frame_count.setRange(0, 0)
        self.dark_frame_count.setValue(0)
        self.dark_frame_count.setFixedWidth(80)
        self.dark_frame_count.setToolTip("Number of random darkest frames to include")
        self.dark_frame_count.frameChanged.connect(self.on_dark_frame_count_changed)

        self.light_frame_count = FrameEdit(frame_count_widget)
        self.light_frame_count.setRange(0, 0)
        self.light_frame_count.setValue(0)
        self.light_frame_count.setFixedWidth(80)
        self.light_frame_count.setToolTip("Number of random lightest frames to include")
        self.light_frame_count.frameChanged.connect(self.on_light_frame_count_changed)

        frame_count_layout.addWidget(self.random_frame_count)
        frame_count_layout.addStretch()
        frame_count_layout.addWidget(QLabel("Dark:"))
        frame_count_layout.addWidget(self.dark_frame_count)
        frame_count_layout.addSpacing(4)
        frame_count_layout.addWidget(QLabel("Light:"))
        frame_count_layout.addWidget(self.light_frame_count)

        auto_select_frame_layout.addRow("Total Random:", frame_count_widget)

        # Range limit
        self.range_mode_selector = SegmentedControl(["Frame", "Time"], auto_select_container)
        self.range_mode_selector.setToolTip("Switch between frame-based and time-based range selection")
        auto_select_frame_layout.addRow("Range Unit:", self.range_mode_selector)

        self.range_stack = QStackedWidget(auto_select_container)

        # Frame/time range inputs
        frame_page = QWidget(self.range_stack)
        frame_row = QHBoxLayout(frame_page)
        frame_row.setContentsMargins(0, 0, 0, 0)
        frame_row.setSpacing(8)

        self.frame_edit_start = FrameEdit(frame_page)
        self.frame_edit_start.setToolTip("Start frame for comparison range")
        self.frame_edit_start.frameChanged.connect(self.on_frame_edit_start_changed)

        self.frame_edit_end = FrameEdit(frame_page)
        self.frame_edit_end.setToolTip("End frame for comparison range")
        self.frame_edit_end.frameChanged.connect(self.on_frame_edit_end_changed)

        frame_row.addWidget(self.frame_edit_start)
        frame_row.addWidget(self.frame_edit_end)
        self.range_stack.addWidget(frame_page)

        time_page = QWidget(self.range_stack)
        time_row = QHBoxLayout(time_page)
        time_row.setContentsMargins(0, 0, 0, 0)
        time_row.setSpacing(8)

        self.time_edit_start = TimeEdit(time_page)
        self.time_edit_start.setToolTip("Start time for comparison range")
        self.time_edit_start.valueChanged.connect(self.on_time_edit_start_changed)

        self.time_edit_end = TimeEdit(time_page)
        self.time_edit_end.setToolTip("End time for comparison range")
        self.time_edit_end.valueChanged.connect(self.on_time_edit_end_changed)

        time_row.addWidget(self.time_edit_start)
        time_row.addWidget(self.time_edit_end)

        self.range_stack.addWidget(time_page)
        self.range_mode_selector.segmentChanged.connect(self.range_stack.setCurrentIndex)

        auto_select_frame_layout.addRow("Range:", self.range_stack)

        # Picture Type Group
        self.frame_props_container = QGroupBox("Frame Properties", auto_select_container)
        frame_props_layout = QHBoxLayout(self.frame_props_container)

        self.pict_type_i_cb = QCheckBox("I-Frame", self.frame_props_container)
        self.pict_type_i_cb.setToolTip("Include I-frames (Intra-coded)")
        self.pict_type_p_cb = QCheckBox("P-Frame", self.frame_props_container)
        self.pict_type_p_cb.setToolTip("Include P-frames (Predictive)")
        self.pict_type_b_cb = QCheckBox("B-Frame", self.frame_props_container)
        self.pict_type_b_cb.setToolTip("Include B-frames (Bi-predictive)")
        self.combed_cb = QCheckBox("Combed", self.frame_props_container)
        self.combed_cb.setToolTip("Include Combed frames")
        self.pict_type_i_cb.setChecked(self.settings.global_.pict_types_i)
        self.pict_type_p_cb.setChecked(self.settings.global_.pict_types_p)
        self.pict_type_b_cb.setChecked(self.settings.global_.pict_types_b)
        self.combed_cb.setChecked(self.settings.global_.combed)
        self.pict_type_i_cb.toggled.connect(lambda state: setattr(self.settings.global_, "pict_types_i", state))
        self.pict_type_p_cb.toggled.connect(lambda state: setattr(self.settings.global_, "pict_types_p", state))
        self.pict_type_b_cb.toggled.connect(lambda state: setattr(self.settings.global_, "pict_types_b", state))
        self.combed_cb.toggled.connect(lambda state: setattr(self.settings.global_, "combed", state))
        frame_props_layout.addWidget(self.pict_type_i_cb)
        frame_props_layout.addWidget(self.pict_type_p_cb)
        frame_props_layout.addWidget(self.pict_type_b_cb)
        frame_props_layout.addWidget(self.combed_cb)

        auto_select_frame_layout.addRow(self.frame_props_container)

        # The actual button
        self.select_frames_btn = QPushButton("Select frames", auto_select_container)
        self.select_frames_btn.setDisabled(True)
        self.select_frames_btn.setToolTip("Collect frames based on the current settings")
        self.select_frames_btn.clicked.connect(self.on_select_frames_clicked)

        auto_select_frame_layout.addRow(self.select_frames_btn)
        form.addRow(auto_select_container)

        self.extract_btn = QPushButton("Extract", self.clip_section)
        self.extract_btn.setDisabled(True)
        self.extract_btn.setToolTip("Extract selected frames to disk")
        self.extract_btn.clicked.connect(self.on_extract_btn_clicked)
        form.addRow(self.extract_btn)

        main.add_section(self.clip_section)

    def _setup_upload_settings(self, main: MainCompWidget) -> None:
        section = Accordion("Upload Settings", main)
        section.setToolTip("Configure upload settings for Slow.pics")
        form = section.add_form_layout()

        # Collection name
        self.collection_name = QLineEdit(section, placeholderText="Name for the image collection")
        self.collection_name.setToolTip("The name of the comparison collection")
        form.addRow("Collection:", self.collection_name)

        # Network stuff
        self.tmdb_name = QLineEdit(section, placeholderText="Search for a movie or TV show...")
        self.tmdb_name.setToolTip("Movie or TV show name to fetch metadata from TMDB")
        form.addRow("TMDB Name:", self.tmdb_name)

        self.tags = QLineEdit(section, placeholderText="Search for additional informations...")
        self.tags.setToolTip("Additional tags to help find the comparison")
        form.addRow("Tags:", self.tags)

        # Public / NSFW checkboxes
        flags_widget = QWidget(section)
        flags_row = QHBoxLayout(flags_widget)
        flags_row.setContentsMargins(0, 0, 0, 0)
        flags_row.setSpacing(16)

        self.public_check = QCheckBox("Public", flags_widget)
        self.public_check.setToolTip("Make the comparison publicly visible")
        self.nsfw_check = QCheckBox("NSFW", flags_widget)
        self.nsfw_check.setToolTip("Mark the comparison as Not Safe For Work")

        flags_row.addWidget(self.public_check)
        flags_row.addWidget(self.nsfw_check)
        flags_row.addStretch(1)

        form.addRow("Flags:", flags_widget)

        # Remove after N days
        self.remove_after = QSpinBox(section)
        self.remove_after.setRange(0, 999999)
        self.remove_after.setSpecialValueText("Never")
        self.remove_after.setSuffix(" days")
        self.remove_after.setToolTip("Remove comparison after N days (0 = never)")
        form.addRow("Auto-Remove:", self.remove_after)

        self.upload_btn = QPushButton("Upload", section)
        self.upload_btn.setToolTip("Upload extracted frames")
        form.addRow(self.upload_btn)

        main.add_section(section)

    def _setup_actions(self, main: MainCompWidget) -> None:
        actions_widget = QWidget(main)
        actions_layout = QVBoxLayout(actions_widget)
        actions_layout.setContentsMargins(8, 4, 8, 4)
        actions_layout.setSpacing(6)

        self.do_all_btn = QPushButton("Extract && Upload", actions_widget)
        self.do_all_btn.setToolTip("Extract selected frames → Upload")
        actions_layout.addWidget(self.do_all_btn)

        main.add_section(actions_widget)

    @cache
    def init_load(self) -> None:
        shortest_output = min(self.api.voutputs, key=lambda v: v.info.total_duration)

        max_frame = shortest_output.info.total_frames - 1
        self.frame_edit_start.setRange(0, max_frame)
        self.frame_edit_end.setRange(0, max_frame)

        self.frame_edit_start.setValue(0)
        self.frame_edit_end.setValue(max_frame)

        qtime_s = Time().to_qtime()
        qtime_e = shortest_output.frame_to_time(max_frame).to_qtime()

        self.time_edit_start.setTime(qtime_s)
        self.time_edit_start.setTimeRange(qtime_s, qtime_e)

        self.time_edit_end.setTime(qtime_e)
        self.time_edit_end.setTimeRange(qtime_s, qtime_e)

    def on_current_voutput_changed(self, voutput: VideoOutputProxy, tab_index: int) -> None:
        self.init_load()

        if self.pict_types_supported and not any("_PictType" in props for props in voutput.props.values()):
            self.pict_types_supported = False
            self.pict_type_i_cb.setDisabled(True)
            self.pict_type_p_cb.setDisabled(True)
            self.pict_type_b_cb.setDisabled(True)

    def on_frame_edit_start_changed(self, new: Frame, old: Frame) -> None:
        self.frame_edit_end.setMinimum(new)
        self.time_edit_start.setTime(self.api.current_voutput.frame_to_time(new).to_qtime())

    def on_frame_edit_end_changed(self, new: Frame, old: Frame) -> None:
        self.frame_edit_start.setMaximum(new)
        self.time_edit_end.setTime(self.api.current_voutput.frame_to_time(new).to_qtime())

    def on_time_edit_start_changed(self, new: QTime, old: QTime) -> None:
        self.time_edit_end.setMinimumTime(new)
        self.frame_edit_start.setValue(self.api.current_voutput.time_to_frame(Time.from_qtime(new)))

    def on_time_edit_end_changed(self, new: QTime, old: QTime) -> None:
        self.time_edit_start.setMaximumTime(new)
        self.frame_edit_end.setValue(self.api.current_voutput.time_to_frame(Time.from_qtime(new)))

    def on_list_size_changed(self, delta: int) -> None:
        self.random_frame_count.setMaximum(clamp(self.random_frame_count.maximum() - delta, 0, 40))
        self.extract_btn.setEnabled(len(self.frames_list.get_data()) > 0)

    def on_random_frame_count_changed(self, new: Frame, old: Frame) -> None:
        dark_val = self.dark_frame_count.value()
        light_val = self.light_frame_count.value()

        if dark_val + light_val > new:
            total = dark_val + light_val
            new_dark = int((dark_val / total) * new)
            new_light = int((light_val / total) * new)

            self.dark_frame_count.setMaximum(new)
            self.light_frame_count.setMaximum(new)

            self.dark_frame_count.setValue(new_dark)
            self.light_frame_count.setValue(new_light)

        self.dark_frame_count.setMaximum(new - self.light_frame_count.value())
        self.light_frame_count.setMaximum(new - self.dark_frame_count.value())

        if new == 0:
            self.select_frames_btn.setDisabled(True)
        else:
            self.select_frames_btn.setEnabled(True)

    def on_dark_frame_count_changed(self, new: Frame, old: Frame) -> None:
        self.light_frame_count.setMaximum(self.random_frame_count.value() - new)

    def on_light_frame_count_changed(self, new: Frame, old: Frame) -> None:
        self.dark_frame_count.setMaximum(self.random_frame_count.value() - new)

    def on_add_multiple_frames(self) -> None:
        text, ok = QInputDialog.getText(
            self,
            "Add Multiple Frames",
            "Enter frames as a Python list or tuple (e.g. [1, 2, 3]):",
        )
        if not any([ok, text]):
            return
        try:
            frames = ast.literal_eval(text)
        except (SyntaxError, ValueError) as e:
            logger.error("Failed to parse frame list: %s", e)
            return

        if isinstance(frames, int):
            frames = [frames]
        elif not isinstance(frames, Sequence):
            logger.error("Invalid frame list format: %s", text)
            return

        for f in frames:
            if f in range(self.api.current_voutput.info.total_frames):
                self.frames_list.add_item(frame=f)
            else:
                logger.warning("Skipping invalid frame number: %s", f)

    def on_select_frames_clicked(self) -> None:
        self.select_frames_btn.setDisabled(True)
        worker = SelectFrameWorker(self.api, self)

        @run_in_loop
        def on_finished(future: Future[list[tuple[Time, FrameSourceProvider]]]) -> None:
            if future.exception():
                return

            times = future.result()

            v = self.api.current_voutput
            for t, src_provider in times:
                self.frames_list.add_item(
                    frame=v.time_to_frame(t),
                    get_pict_type=worker.should_check_pict,
                    src_provider=src_provider,
                )

            self.select_frames_btn.setEnabled(True)
            self.progress_bar.reset_progress()

            self._pending_select_frames = None
            worker.deleteLater()

        self._pending_select_frames = worker.run()
        self._pending_select_frames.add_done_callback(on_finished)

    def on_extract_btn_clicked(self) -> None:
        @run_in_loop
        def prepare_and_extract(future: Future[Any] | None = None) -> None:
            if future and future.exception():
                return

            data = self.frames_list.get_data()
            included_outputs = self.outputs_dropdown.included_outputs

            self.progress_bar.update_progress(
                range=(0, len(data) * len(included_outputs)),
                fmt="Extracting frames %v / %m",
                value=0,
            )

            self._pending_extract_frames = self.start_extraction_task(data, included_outputs)

            @run_in_loop
            def on_finished(*_: Any) -> None:
                self.clip_section.setEnabled(True)
                self.progress_bar.reset_progress()
                self._pending_extract_frames = None

            self._pending_extract_frames.add_done_callback(on_finished)

        self.clip_section.setDisabled(True)

        if self._pending_select_frames:
            self._pending_select_frames.add_done_callback(prepare_and_extract)
        else:
            prepare_and_extract()

    @run_in_background(name="ExtractFrames")
    def start_extraction_task(self, data: list[tuple[Time, str]], included_outputs: list[int]) -> None:
        if not (storage := self.api.get_local_storage(self)):
            raise NotImplementedError

        path = storage / str(datetime.now())
        workers = list[Future[None]]()

        with self.api.vs_context():
            is_fpng_available = hasattr(core, "fpng")

            for output in self.api.voutputs:
                if output.vs_index not in included_outputs:
                    continue

                images_path = path / f"({output.vs_index}) ({output.vs_name})"
                images_path = sanitize_filepath(images_path, replacement_text="_")
                images_path.mkdir(parents=True, exist_ok=True)

                clip = self.api.packer.to_rgb_planar(output.vs_output.clip, format=RGB24)
                frames = [output.time_to_frame(t) for t, _ in data]
                clip_image_path = images_path / f"%0{ndigits(max(frames))}d.png"

                if is_fpng_available:
                    f = self._fpng_extract(clip, clip_image_path, frames)
                else:
                    f = self._qt_extract(clip, clip_image_path, frames)

                workers.append(f)

            # Wait for workers to finish extracting
            wait(workers)

    @run_in_background(name="ExtractFPNG")
    def _fpng_extract(self, clip: VideoNode, path: Path, frames: Sequence[int]) -> None:
        # TODO: Maybe add alpha support?
        with self.api.vs_context():
            # 1 - slow compression (smaller output file)
            clip = clip.fpng.Write(filename=str(path), compression=1)
            remapped = remap_frames(clip, frames)

            clip_async_render(remapped, progress=lambda *_: self.progress_bar.update_progress(increment=1))

    @run_in_background(name="ExtractQt")
    def _qt_extract(self, clip: VideoNode, path: Path, frames: Sequence[int]) -> None:
        with self.api.vs_context():
            clip = self.api.packer.to_rgb_packed(clip)
            remapped = remap_frames(clip, frames)

            workers = list[Future[None]]()

            for n, vs_frame in zip(frames, remapped.frames(close=True)):
                qimage = self.api.packer.frame_to_qimage(vs_frame).copy()
                f = self._qt_save(qimage, path.with_stem(path.stem % n))
                workers.append(f)

            wait(workers)

    @run_in_background(name="QtSave")
    def _qt_save(self, qimage: QImage, path: Path) -> None:
        qimage.save(str(path), "PNG", 75)  # type: ignore[call-overload]
        self.progress_bar.update_progress(increment=1)


class SelectFrameWorker(QObject):
    ALLOWED_FRAME_SEARCHES = 150

    def __init__(self, api: PluginAPI, parent: CompPlugin) -> None:
        super().__init__(parent)
        self.api = api
        self.progress_bar = parent.progress_bar

        self.start = Time.from_qtime(parent.time_edit_start.time())
        self.end = Time.from_qtime(parent.time_edit_end.time())
        self.dark = parent.dark_frame_count.value()
        self.light = parent.light_frame_count.value()
        self.normal = parent.random_frame_count.value() - self.dark - self.light

        # Existing frames to avoid duplicates
        v = self.api.current_voutput
        self.checked = [int(v.time_to_frame(t)) for t, _ in parent.frames_list.get_data()]

        # Picture types
        self.pict_types = list[str]()
        if parent.pict_type_i_cb.isChecked():
            self.pict_types.append("I")
        if parent.pict_type_p_cb.isChecked():
            self.pict_types.append("P")
        if parent.pict_type_b_cb.isChecked():
            self.pict_types.append("B")

        self.should_check_pict = len(self.pict_types) < 3 and parent.pict_types_supported
        self.should_check_combed = not parent.combed_cb.isChecked()

    @run_in_background(name="SelectFrames")
    def run(self) -> list[tuple[Time, FrameSourceProvider]]:
        with self.api.vs_context():
            return self.get()

    def get(self) -> list[tuple[Time, FrameSourceProvider]]:
        found_times = list[tuple[Time, FrameSourceProvider]]()

        if self.normal > 0:
            found_times.extend((t, FrameSourceProvider.RANDOM) for t in self._get_normal_frames())

        if self.dark > 0 or self.light > 0:
            found_times.extend(self._get_light_dark_frames())

        return sorted(set(found_times))

    def _get_normal_frames(self) -> list[Time]:
        v = self.api.current_voutput
        start_frame, end_frame = v.time_to_frame(self.start), v.time_to_frame(self.end)

        self.progress_bar.update_progress(range=(0, self.normal), fmt="Selecting frames %v / %m", value=0)

        random_frames = list[Time]()
        base_clip = core.std.BlankClip(width=1, height=1, format=GRAY8, length=len(self.api.voutputs), keep=True)
        other_clips = [source.vs_output.clip for source in self.api.voutputs]

        while len(random_frames) < self.normal:
            for _ in range(self.ALLOWED_FRAME_SEARCHES):
                rnum = get_random_number_interval(
                    start_frame,
                    end_frame,
                    self.normal,
                    len(random_frames),
                    self.checked,
                )
                self.checked.append(rnum)

                is_valid = True
                if self.should_check_pict or self.should_check_combed:
                    # Check frame properties across all outputs
                    node_frames = core.std.FrameEval(
                        base_clip,
                        lambda n, r=rnum: base_clip.std.CopyFrameProps(
                            other_clips[n][r], props=["_PictType", "_Combed"]
                        ),
                    )

                    for f in node_frames.frames(close=True):
                        is_pict_type_not_selected = (
                            self.should_check_pict
                            and get_prop(f, "_PictType", str, default="", func="__vsview__") not in self.pict_types
                        )
                        is_combed = (
                            self.should_check_combed
                            and get_prop(f, "_Combed", int, default=0, func="__vsview__")  # No format
                        )

                        if is_pict_type_not_selected or is_combed:
                            is_valid = False
                            break

                if is_valid:
                    random_frames.append(v.frame_to_time(rnum))
                    self.progress_bar.update_progress(value=len(random_frames))
                    break
            else:
                logger.warning(
                    "Max attempts reached searching for random frames. Found %s/%s",
                    len(random_frames),
                    self.normal,
                )
                break

        return random_frames

    def _get_light_dark_frames(self) -> list[tuple[Time, FrameSourceProvider]]:
        v = self.api.current_voutput
        start, end = v.time_to_frame(self.start), v.time_to_frame(self.end)

        # Sample frames for brightness analysis
        step = max(1, (end - start) // (self.ALLOWED_FRAME_SEARCHES * 3))
        frames_to_check = range(start, end, step)

        self.progress_bar.update_progress(
            range=(0, len(frames_to_check)), fmt="Checking frames light levels %v / %m", value=0
        )

        checked_count = 0

        def _progress(*_: Any) -> None:
            nonlocal checked_count
            checked_count += 1
            self.progress_bar.update_progress(value=checked_count)

        decimated = remap_frames(v.vs_output.clip, frames_to_check).std.PlaneStats()
        avg_levels = clip_data_gather(
            decimated,
            _progress,
            lambda n, f: get_prop(f, "PlaneStatsAverage", float, default=0, func=self._get_light_dark_frames),
        )

        # Pair levels with frames and sort by brightness
        sorted_frames = [f for _, f in sorted(zip(avg_levels, frames_to_check))]

        dark = sorted_frames[: self.dark] if self.dark else []
        light = sorted_frames[-self.light :] if self.light else []

        return [
            *((v.frame_to_time(f), FrameSourceProvider.RANDOM_DARK) for f in dark),
            *((v.frame_to_time(f), FrameSourceProvider.RANDOM_LIGHT) for f in light),
        ]
