from collections.abc import Sequence
from concurrent.futures import Future, wait
from datetime import datetime
from functools import cache
from logging import getLogger
from pathlib import Path
from typing import Annotated

from jetpytools import clamp, ndigits
from pathvalidate import sanitize_filepath
from pydantic import BaseModel
from PySide6.QtCore import Qt, QTime
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from vapoursynth import RGB24, VideoNode
from vsengine.loops import get_loop
from vstools import clip_async_render, core, remap_frames

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

from .ui import FrameThumbnailList, MainCompWidget, OutputDropdown

logger = getLogger(__name__)


PLUGIN_ID = "jet_vsview_comp"
PLUGIN_DISPLAY = "Comparison"
COOKIE_KEY = "cookies.v1"


class GlobalSettings(BaseModel):
    login: Annotated[str, Login(label="Slow.pics credentials", namespace=PLUGIN_ID)]


    pict_types_i: bool = True
    pict_types_p: bool = True
    pict_types_b: bool = True
    public_comp_default: bool = True


class CompPlugin(WidgetPluginBase[GlobalSettings, None], IconReloadMixin):
    identifier = PLUGIN_ID
    display_name = PLUGIN_DISPLAY

    shortcuts = (ActionDefinition(f"{PLUGIN_ID}.add_current_frame", "Add Current Frame", "Ctrl+Space"),)

    def __init__(self, parent: QWidget, api: PluginAPI) -> None:
        super().__init__(parent, api)

        self.pict_types_supported = True

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        with MainCompWidget(self) as main:
            self._setup_clip_options(main)
            self._setup_upload_settings(main)
            self._setup_actions(main)

        main_layout.addWidget(main)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimumHeight(24)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setHidden(True)
        self.progress_bar.setToolTip("Overall progress of extraction and upload tasks")
        main_layout.addWidget(self.progress_bar)

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
        toolbar.addAction(self.add_frame_act)

        self.remove_frame_act = self.make_action(IconName.MINUS, "Remove selected frame(s) from the list", frame_widget)
        self.remove_frame_act.setIconText("Remove selected frame(s)")
        self.remove_frame_act.setEnabled(False)

        toolbar.addActions([self.add_frame_act, self.remove_frame_act])

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

        self.add_frame_act.triggered.connect(self.frames_list.add_item)
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
        self.picture_type_container = QGroupBox("Picture Type", auto_select_container)
        pict_type_layout = QHBoxLayout(self.picture_type_container)

        self.pict_type_i_cb = QCheckBox("I-Frame", self.picture_type_container)
        self.pict_type_i_cb.setToolTip("Include I-frames (Intra-coded)")
        self.pict_type_p_cb = QCheckBox("P-Frame", self.picture_type_container)
        self.pict_type_p_cb.setToolTip("Include P-frames (Predictive)")
        self.pict_type_b_cb = QCheckBox("B-Frame", self.picture_type_container)
        self.pict_type_b_cb.setToolTip("Include B-frames (Bi-predictive)")
        self.pict_type_i_cb.setChecked(self.settings.global_.pict_types_i)
        self.pict_type_p_cb.setChecked(self.settings.global_.pict_types_p)
        self.pict_type_b_cb.setChecked(self.settings.global_.pict_types_b)
        self.pict_type_i_cb.toggled.connect(lambda state: setattr(self.settings.global_, "pict_types_i", state))
        self.pict_type_p_cb.toggled.connect(lambda state: setattr(self.settings.global_, "pict_types_p", state))
        self.pict_type_b_cb.toggled.connect(lambda state: setattr(self.settings.global_, "pict_types_b", state))
        pict_type_layout.addWidget(self.pict_type_i_cb)
        pict_type_layout.addWidget(self.pict_type_p_cb)
        pict_type_layout.addWidget(self.pict_type_b_cb)

        auto_select_frame_layout.addRow(self.picture_type_container)

        # The actual button
        self.select_frames_btn = QPushButton("Select frames", auto_select_container)
        self.select_frames_btn.setToolTip("Collect frames based on the current settings")
        self.select_frames_btn.clicked.connect(self.on_select_frames_clicked)

        auto_select_frame_layout.addRow(self.select_frames_btn)
        form.addRow(auto_select_container)

        self.extract_btn = QPushButton("Extract", self.clip_section)
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
            self.picture_type_container.setEnabled(False)

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

    def on_dark_frame_count_changed(self, new: Frame, old: Frame) -> None:
        self.light_frame_count.setMaximum(self.random_frame_count.value() - new)

    def on_light_frame_count_changed(self, new: Frame, old: Frame) -> None:
        self.dark_frame_count.setMaximum(self.random_frame_count.value() - new)

    def on_select_frames_clicked(self) -> None:
        # TODO
        ...

    def on_extract_btn_clicked(self) -> None:
        self.clip_section.setDisabled(True)
        f = self.start_extraction_task(self.frames_list.get_data())
        f.add_done_callback(lambda _: self.clip_section.setEnabled(True))

    @run_in_background(name="ExtractFrames")
    def start_extraction_task(self, data: list[tuple[Time, str]]) -> None:
        if not (storage := self.api.get_local_storage(self)):
            raise NotImplementedError

        path = storage / str(datetime.now())
        workers = list[Future[None]]()

        with self.api.vs_context():
            is_fpng_available = hasattr(core, "fpng")

            for output in self.api.voutputs:
                if output.vs_index not in self.outputs_dropdown.included_outputs:
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

            # TODO: add progress bar callbacks
            clip_async_render(remapped, progress="Progress...")

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
