from collections.abc import Sequence
from concurrent.futures import Future, wait
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jetpytools import ndigits
from pathvalidate import sanitize_filepath
from PySide6.QtGui import QImage
from vapoursynth import GRAY8, RGB24, VideoNode
from vstools import clip_async_render, clip_data_gather, core, get_prop, remap_frames

from vsview.api import PluginAPI, Time, run_in_background

from .ui import FrameSourceProvider
from .utils import get_random_number_interval

if TYPE_CHECKING:
    from .plugin import CompPlugin

logger = getLogger(__name__)


class ExtractFramesWorker:
    def __init__(self, api: PluginAPI, parent: CompPlugin) -> None:
        self.api = api
        self.progress_bar = parent.progress_bar
        self.data = parent.frames_list.get_data()
        self.included_outputs = parent.outputs_dropdown.included_outputs

        if not (storage := self.api.get_local_storage(parent)):
            raise NotImplementedError
        self.storage = storage

    @run_in_background(name="ExtractFrames")
    def run(self) -> None:
        self.progress_bar.update_progress(
            range=(0, len(self.data) * len(self.included_outputs)),
            fmt="Extracting frames %v / %m",
            value=0,
        )

        path = self.storage / str(datetime.now())
        workers = list[Future[None]]()

        with self.api.vs_context():
            is_fpng_available = hasattr(core, "fpng")

            for output in self.api.voutputs:
                if output.vs_index not in self.included_outputs:
                    continue

                images_path = path / f"({output.vs_index}) ({output.vs_name})"
                images_path = sanitize_filepath(images_path, replacement_text="_")
                images_path.mkdir(parents=True, exist_ok=True)

                clip = self.api.packer.to_rgb_planar(output.vs_output.clip, format=RGB24)
                frames = [output.time_to_frame(t) for t, _ in self.data]
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


class SelectFrameWorker:
    ALLOWED_FRAME_SEARCHES = 150

    def __init__(self, api: PluginAPI, parent: CompPlugin) -> None:
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
