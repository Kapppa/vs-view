"""RGB packing implementations for VapourSynth to Qt conversion."""

from __future__ import annotations

import sys
from abc import ABC
from collections.abc import Mapping
from enum import IntEnum
from logging import Filter, LogRecord, getLogger
from typing import Any, Literal, override

import vapoursynth as vs
from PySide6.QtGui import QImage
from vspackrgb.helpers import get_plane_buffer, packrgb

from .settings import SettingsManager

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

logger = getLogger(__name__)


class FramePropsFilter(Filter):
    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self.msgs = set[str]()

    @override
    def filter(self, record: LogRecord) -> bool | LogRecord:
        logged = super().filter(record)

        if logged and record.msg not in self.msgs:
            self.msgs.add(record.msg)
            return logged
        return False


logger.addFilter(FramePropsFilter(logger.name))


def select_in_matrix(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
    if f.format.color_family == vs.RGB and f.props.get("_Matrix", vs.MATRIX_UNSPECIFIED) == vs.MATRIX_UNSPECIFIED:
        f = f.copy()
        f.props["_Matrix"] = vs.MATRIX_RGB
    return f


def warn_missing_props(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
    specs: list[IntEnum] = [
        vs.MatrixCoefficients(f.props.get("_Matrix", 2)),
        vs.ColorPrimaries(f.props.get("_Primaries", 2)),
        vs.TransferCharacteristics(f.props.get("_Transfer", 2)),
    ]

    if unknowns := [spec for spec in specs if spec == 2]:
        prop_names = [e.name.split("_")[0].title() for e in unknowns]
        logger.warning("The following properties are missing: %r", prop_names)

    return f


class Packer(ABC):
    """RGB packer"""

    FORMAT_CONFIG: Mapping[int, tuple[vs.PresetVideoFormat, vs.PresetVideoFormat, QImage.Format, QImage.Format]] = {
        8: (vs.RGB24, vs.GRAY8, QImage.Format.Format_RGB32, QImage.Format.Format_ARGB32),
        10: (vs.RGB30, vs.GRAY10, QImage.Format.Format_RGB30, QImage.Format.Format_A2RGB30_Premultiplied),
    }

    def __init__(self, bit_depth: int | None = None) -> None:
        self.bit_depth = bit_depth or SettingsManager.global_settings.view.bit_depth
        self.vs_format, self.vs_aformat, self.qt_format, self.qt_aformat = Packer.FORMAT_CONFIG[self.bit_depth]

    def to_rgb_planar(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        """Converts clip to planar vs.RGB24 or vs.RGB30."""

        params = dict[str, Any](
            format=self.vs_format,
            dither_type=SettingsManager.global_settings.view.dither_type,
            resample_filter_uv=SettingsManager.global_settings.view.chroma_resizer.vs_func,
            filter_param_a_uv=SettingsManager.global_settings.view.chroma_resizer.param_a,
            filter_param_b_uv=SettingsManager.global_settings.view.chroma_resizer.param_b,
            transfer=vs.TRANSFER_BT709,
            primaries=vs.PRIMARIES_BT709,
        )

        # Returns directly the clip without checking anything.
        # If color specs are set, this will work.
        if (policy := SettingsManager.global_settings.view.props_policy) == "error":
            return clip.resize.Point(**params | kwargs)

        if policy == "warn":
            clip = clip.std.ModifyFrame(clip, warn_missing_props)

        # If the corresponding frameprop is set to a value other than unspecified,
        # the frameprop is used instead of this parameter
        in_params = dict[str, Any](transfer_in=vs.TRANSFER_BT709, primaries_in=vs.PRIMARIES_BT709)
        if clip.format.id == vs.PresetVideoFormat.NONE:
            clip = clip.std.ModifyFrame(clip, select_in_matrix)
        elif clip.format.color_family is vs.RGB:
            in_params["matrix_in"] = vs.MATRIX_RGB

        return clip.resize.Point(**params | in_params | kwargs)

    def to_rgb_packed(self, clip: vs.VideoNode, alpha: vs.VideoNode | Literal[True] | None = None) -> vs.VideoNode:
        """Converts planar vs.RGB24 or vs.RGB30 to interleaved BGRA32 or RGB30 to packed A2R10G10B10"""
        return packrgb(clip, alpha, "cython")

    def pack_clip(self, clip: vs.VideoNode, alpha: vs.VideoNode | Literal[True] | None = None) -> vs.VideoNode:
        """Converts a planar VideoNode and an optional alpha mask to a packed RGB/RGBA VideoNode."""
        if isinstance(alpha, vs.VideoNode):
            alpha = alpha.resize.Point(
                format=self.vs_aformat,
                dither_type=SettingsManager.global_settings.view.dither_type,
            )

        planar = self.to_rgb_planar(clip)
        packed = self.to_rgb_packed(planar, alpha)

        return packed.std.SetFrameProp("VSViewHasAlpha", True) if alpha else packed

    def frame_to_qimage(self, frame: vs.VideoFrame, **kwargs: Any) -> QImage:
        """
        Wraps a packed VapourSynth VideoFrame into a QImage.

        If the `copy_qimage` setting is enabled, ownership of the memory is transferred to Qt
        by returning a copy of the image.
        Otherwise, the returned QImage **does not own its memory** and points directly
        to the VapourSynth frame's buffer.

        !!! warning
            When `copy_qimage` is disabled, you MUST either keep the source `frame` alive as long
            as the QImage is used, or call ``.copy()`` on the returned QImage.
        """

        alpha = "VSViewHasAlpha" in frame.props or "_Alpha" in frame.props

        params = dict[str, Any](format=self.qt_aformat if alpha else self.qt_format) | kwargs

        # QImage supports Buffer inputs
        img = QImage(
            get_plane_buffer(frame, 0),  # type: ignore[call-overload]
            frame.width,
            frame.height,
            frame.get_stride(0),
            params.pop("format"),
            **params,
        )

        if SettingsManager.global_settings.view.copy_qimage:
            return img.copy()

        return img


@deprecated("Deprecated method. Use Packer.", category=DeprecationWarning)
def get_packer(method: str | None = None, bit_depth: int | None = None) -> Packer:
    return Packer(bit_depth)
