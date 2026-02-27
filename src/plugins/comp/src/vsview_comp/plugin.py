import ast
from collections.abc import Sequence
from concurrent.futures import Future
from functools import cache
from logging import getLogger
from typing import Annotated, Any

from jetpytools import clamp
from pydantic import BaseModel
from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt, QTime, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from vsview.api import (
    Accordion,
    ActionDefinition,
    Frame,
    FrameEdit,
    IconName,
    IconReloadMixin,
    Login,
    PluginAPI,
    SegmentedControl,
    Time,
    TimeEdit,
    VideoOutputProxy,
    WidgetPluginBase,
    run_in_loop,
)

from .models import TMDBTitle
from .ui import (
    FrameSourceProvider,
    FrameThumbnailList,
    MainCompWidget,
    OutputDropdown,
    ProgressBar,
    PushButton,
    TagsLineEdit,
    TagsListPopup,
    TMDBListPopup,
)
from .worker import ExtractFramesWorker, SelectFrameWorker, SlowPicsWorker, Tag, TMDBWorker

logger = getLogger(__name__)


PLUGIN_ID = "jet_vsview_comp"
PLUGIN_DISPLAY = "Comparison"
LOGIN_CONTEXT = "slowpicslogin"
COOKIE_KEY = "cookies.v1"


class GlobalSettings(BaseModel):
    login: Annotated[
        str,
        Login(
            label="Slow.pics credentials",
            namespace=PLUGIN_ID,
            context=LOGIN_CONTEXT,
            tooltip="The Slowpoke.pics credentials for login",
        ),
    ]

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
        self._pending_tags: Future[list[Tag]] | None = None

        # Build UI
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

        self.api.register_action(
            f"{PLUGIN_ID}.add_current_frame",
            self.add_frame_act,
            context=Qt.ShortcutContext.WindowShortcut,
        )

        self.api.register_on_destroy(self.init_load.cache_clear)

        # Disable the whole plugin if we don't have a local storage
        self.setEnabled(has_file := bool(self.api.file_path))
        if not has_file:
            self.setToolTip("Plugin disabled: the current workspace doesn't have a local storage.")

        # Install event filter to override child tooltips when disabled
        if not (app := QApplication.instance()):
            raise SystemError

        app.installEventFilter(self)
        self.destroyed.connect(lambda: app.removeEventFilter(self) if (app := QApplication.instance()) else None)

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

        self.extract_btn = PushButton("Extract", self.clip_section)
        self.extract_btn.setDisabled(True)
        self.extract_btn.setToolTip("Extract selected frames to disk")
        self.extract_btn.clicked.connect(self.on_extract_btn_clicked)
        self.extract_btn.enabledChanged.connect(lambda enabled: self.upload_btn.setDisabled(enabled))
        form.addRow(self.extract_btn)

        main.add_section(self.clip_section)

    def _setup_upload_settings(self, main: MainCompWidget) -> None:
        section = Accordion("Upload Settings", main)
        section.setToolTip("Configure upload settings for Slow.pics")
        form = section.add_form_layout()

        self.slowpics_worker = SlowPicsWorker(self.secrets)

        # Collection name
        self.collection_name = QLineEdit(section, placeholderText="Name for the image collection")
        self.collection_name.setToolTip("The name of the comparison collection")
        form.addRow("Collection:", self.collection_name)

        # Network stuff
        self.tmdb_title: TMDBTitle | None = None
        self.tmdb_worker = TMDBWorker()

        self.tmdb_name = QLineEdit(section, placeholderText="Search for a movie or TV show...")
        self.tmdb_name.setToolTip("Movie or TV show name to fetch metadata from TMDB")
        self.tmdb_name.textChanged.connect(self.on_tmdb_text_changed)
        self.tmdb_name.editingFinished.connect(self.on_tmdb_editing_finished)
        form.addRow("TMDB Name:", self.tmdb_name)

        self.tmdb_popup = TMDBListPopup(section, self.tmdb_name)
        self.tmdb_popup.itemClicked.connect(self.on_tmdb_item_selected)
        self.tmdb_popup.itemActivated.connect(self.on_tmdb_item_selected)

        self.tmdb_debounce_timer = QTimer(self, singleShot=True, interval=300)
        self.tmdb_debounce_timer.timeout.connect(self.perform_tmdb_search)

        self.tags = TagsLineEdit(section, placeholder_text="Search for additional informations...")
        self.tags.editingStarted.connect(self.on_tags_editing_started)
        self.tags.inputTextChanged.connect(self.on_tags_input_changed)
        self.tags.tagsChanged.connect(self.on_tags_input_changed)
        self.tags.setToolTip("Additional tags to help find the comparison")
        form.addRow("Tags:", self.tags)

        self.tags_popup = TagsListPopup(section, self.tags)
        self.tags_popup.tagPicked.connect(self.on_tag_item_selected)

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
        self.upload_btn.setDisabled(True)
        self.upload_btn.setToolTip("Upload extracted frames")
        form.addRow(self.upload_btn)

        main.add_section(section)

    def _setup_actions(self, main: MainCompWidget) -> None:
        actions_widget = QWidget(main)
        actions_layout = QVBoxLayout(actions_widget)
        actions_layout.setContentsMargins(8, 4, 8, 4)
        actions_layout.setSpacing(6)

        self.do_all_btn = QPushButton("Extract && Upload", actions_widget)
        self.do_all_btn.setDisabled(True)
        self.do_all_btn.setToolTip("Extract selected frames → Upload")
        actions_layout.addWidget(self.do_all_btn)

        main.add_section(actions_widget)

    @property
    @run_in_loop(return_future=False)
    def pict_types_supported(self) -> bool:
        return self.pict_type_i_cb.isEnabled() and self.pict_type_p_cb.isEnabled() and self.pict_type_b_cb.isEnabled()

    @pict_types_supported.setter
    @run_in_loop(return_future=False)
    def pict_types_supported(self, value: bool) -> None:
        self.pict_type_i_cb.setEnabled(value)
        self.pict_type_p_cb.setEnabled(value)
        self.pict_type_b_cb.setEnabled(value)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # Intercept tooltips for this plugin tree when disabled
        if (
            event.type() == QEvent.Type.ToolTip
            and not self.isEnabled()
            and isinstance(watched, QWidget)
            and (watched is self or self.isAncestorOf(watched))
        ):
            QToolTip.showText(QCursor.pos(), self.toolTip(), watched)
            return True
        return super().eventFilter(watched, event)

    @cache
    @run_in_loop(return_future=False)
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
        self.upload_btn.setEnabled(False)
        self.do_all_btn.setEnabled(len(self.frames_list.get_data()) > 0)

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

        self._pending_select_frames = worker.run()
        self._pending_select_frames.add_done_callback(on_finished)

    def on_extract_btn_clicked(self) -> None:
        @run_in_loop
        def prepare_and_extract(future: Future[Any] | None = None) -> None:
            if future and future.exception():
                return

            worker = ExtractFramesWorker(self.api, self)
            self._pending_extract_frames = worker.run()

            @run_in_loop
            def on_finished(*_: Any) -> None:
                self.extract_btn.setDisabled(True)
                self.clip_section.setEnabled(True)
                self.progress_bar.reset_progress()
                self._pending_extract_frames = None

            self._pending_extract_frames.add_done_callback(on_finished)

        self.clip_section.setDisabled(True)

        if self._pending_select_frames:
            self._pending_select_frames.add_done_callback(prepare_and_extract)
        else:
            prepare_and_extract()

    def on_tmdb_text_changed(self, text: str) -> None:
        self.tmdb_title = None
        if not text.strip():
            self.tmdb_debounce_timer.stop()
            self.tmdb_popup.hide()
            return

        self.tmdb_debounce_timer.start()

    def on_tmdb_editing_finished(self) -> None:
        text = self.tmdb_name.text().strip()

        if not text:
            self.tmdb_title = None
            self.tmdb_popup.hide()
            return

        if self.tmdb_title and self.tmdb_title.name == text:
            return

        with QSignalBlocker(self.tmdb_name):
            self.tmdb_name.clear()

        self.tmdb_title = None
        self.tmdb_popup.hide()

    def perform_tmdb_search(self) -> None:
        if not (query := self.tmdb_name.text().strip()):
            self.tmdb_popup.hide()
            return

        @run_in_loop
        def on_finished(future: Future[list[TMDBTitle]]) -> None:
            if future.exception():
                self.tmdb_popup.hide()
                return

            if self.tmdb_name.text().strip() != query:
                return

            results = future.result()

            if not results:
                self.tmdb_popup.show_no_results()
            else:
                self.tmdb_popup.show_results(results)

        future = self.tmdb_worker.search(query)
        future.add_done_callback(on_finished)

    def on_tmdb_item_selected(self, item: QListWidgetItem) -> None:
        title: TMDBTitle = item.data(Qt.ItemDataRole.UserRole)

        self.tmdb_popup.hide()

        with QSignalBlocker(self.tmdb_name):
            self.tmdb_name.setText(title.name)

        self.tmdb_title = title

    def on_tags_editing_started(self) -> None:
        if self.tags_popup.has_tags():
            self.on_tags_input_changed()
            return

        if self._pending_tags:
            return

        @run_in_loop
        def on_finished(future: Future[list[Tag]]) -> None:
            self._pending_tags = None
            if future.exception():
                self.tags_popup.hide()
                return

            self.tags_popup.set_tags(future.result())
            self.on_tags_input_changed()

        self._pending_tags = self.slowpics_worker.get_tags()
        self._pending_tags.add_done_callback(on_finished)

    def on_tags_input_changed(self, *_: Any) -> None:
        self.tags_popup.show_filtered(self.tags.input_text(), set(self.tags.selected_tags()))

    def on_tag_item_selected(self, value: str, label: str) -> None:
        self.tags.add_tag(value, label)
        self.tags.clear()
        self.on_tags_input_changed()
